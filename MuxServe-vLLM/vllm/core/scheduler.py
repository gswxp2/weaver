import enum
import time
from typing import Dict, Iterable, List, Optional, Tuple, Union

from muxserve.constants import PREEMPT_REQ_NAME_FMT
from muxserve.shm_utils import dump_to_shared_var

from vllm.config import CacheConfig, SchedulerConfig, ModelConfig, ParallelConfig
from vllm.core.block_manager import BlockSpaceManager
from vllm.core.policy import PolicyFactory
from vllm.logger import init_logger
from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
                           SequenceGroupMetadata, SequenceStatus)
from vllm.zmq_tool import ZMQClient

logger = init_logger(__name__)


class PreemptionMode(enum.Enum):
    """Preemption modes.

    1. Swapping: Swap out the blocks of the preempted sequences to CPU memory
    and swap them back in when the sequences are resumed.
    2. Recomputation: Discard the blocks of the preempted sequences and
    recompute them when the sequences are resumed, treating the sequences as
    new prompts.
    """
    SWAP = enum.auto()
    RECOMPUTE = enum.auto()
    ABORT = enum.auto()


class SchedulerOutputs:

    def __init__(
        self,
        scheduled_seq_groups: List[SequenceGroup],
        prompt_run: bool,
        num_batched_tokens: int,
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
        ignored_seq_groups: List[SequenceGroup],
    ) -> None:
        self.scheduled_seq_groups = scheduled_seq_groups
        self.prompt_run = prompt_run
        self.num_batched_tokens = num_batched_tokens
        self.blocks_to_swap_in = blocks_to_swap_in
        self.blocks_to_swap_out = blocks_to_swap_out
        self.blocks_to_copy = blocks_to_copy
        # Swap in and swap out should never happen at the same time.
        assert not (blocks_to_swap_in and blocks_to_swap_out)
        self.ignored_seq_groups = ignored_seq_groups

    def is_empty(self) -> bool:
        # NOTE: We do not consider the ignored sequence groups.
        return (not self.scheduled_seq_groups and not self.blocks_to_swap_in
                and not self.blocks_to_swap_out and not self.blocks_to_copy)


class Scheduler:

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        tcp_client: Optional[ZMQClient] = None,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.model_config = model_config
        self.tcp_client = tcp_client

        self.prompt_limit = min(self.scheduler_config.max_model_len,
                                self.scheduler_config.max_num_batched_tokens)

        # Instantiate the scheduling policy.
        self.policy = PolicyFactory.get_policy(policy_name="fcfs")
        # Create the block space manager.
        self.block_manager = BlockSpaceManager(
            model_name=self.model_config.model_name,
            num_layers=self.model_config.get_num_layers(parallel_config),
            num_heads=self.model_config.get_num_kv_heads(parallel_config),
            block_size=self.cache_config.block_size,
            num_gpu_blocks=self.cache_config.num_gpu_blocks,
            num_cpu_blocks=self.cache_config.num_cpu_blocks,
            sliding_window=self.cache_config.sliding_window,
            tcp_client=self.tcp_client)

        if self.tcp_client is not None:
            self.preempt_req_shm_name = PREEMPT_REQ_NAME_FMT.format(
                self.model_config.model_name, self.model_config.mps_percentage
            )

        # TODO(zhuohan): Use deque instead of list for better performance.
        # Sequence groups in the WAITING state.
        self.waiting: List[SequenceGroup] = []
        # Sequence groups in the RUNNING state.
        self.running: List[SequenceGroup] = []
        # Sequence groups in the SWAPPED state.
        self.swapped: List[SequenceGroup] = []
        # Sequence groups in execution
        self.executing: List[SequenceGroup] = []

        self.requests_in_queue: List[str] = set()

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(seq_group)
        self.requests_in_queue.add(seq_group.request_id)

    def abort_seq_group(self, request_id: Union[str, Iterable[str]]) -> None:
        if isinstance(request_id, str):
            request_id = (request_id, )
        request_ids = set(request_id)
        for state_queue in [self.waiting, self.running, self.swapped]:
            # We need to reverse the list as we are removing elements
            # from it as we iterate over it. If we don't do it,
            # indices will get messed up and we will skip over elements.
            for seq_group in reversed(state_queue):
                if seq_group.request_id in request_ids:
                    # Remove the sequence group from the state queue.
                    state_queue.remove(seq_group)
                    for seq in seq_group.get_seqs():
                        if seq.is_finished():
                            continue
                        seq.status = SequenceStatus.FINISHED_ABORTED
                        self.free_seq(seq)
                    request_ids.remove(seq_group.request_id)
                    self.requests_in_queue.remove(seq_group.request_id)
                    if not request_ids:
                        return

    def release_seq_group(self, request_id: Union[str, Iterable[str]], preempted=False) -> None:
        if isinstance(request_id, str):
            request_id = (request_id, )
        request_ids = set(request_id)
        for state_queue in [self.waiting, self.running, self.swapped, self.executing]:
            # We need to reverse the list as we are removing elements
            # from it as we iterate over it. If we don't do it,
            # indices will get messed up and we will skip over elements.
            for seq_group in reversed(state_queue):
                if seq_group.request_id in request_ids:
                    # Remove the sequence group from the state queue.
                    state_queue.remove(seq_group)
                    for seq in seq_group.get_seqs():
                        seq.status = SequenceStatus.FINISHED_STOPPED
                        self.free_seq(seq)
                    request_ids.remove(seq_group.request_id)
                    self.requests_in_queue.remove(seq_group.request_id)
                    if not request_ids:
                        return

    def has_unfinished_seqs(self) -> bool:
        return self.waiting or self.running or self.swapped

    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped)

    def execute_seq_group(self, request_id: Union[str, List[str]]) -> None:
        if isinstance(request_id, str):
            request_id = [request_id]

        running = []
        request_ids = set(request_id)
        index = 0
        while index < len(self.executing):
            seq_group = self.executing[index]
            if seq_group.request_id in request_ids:
                running.append(seq_group)
                self.executing.pop(index)
            else:
                index += 1

        self.running = running + self.running

    def _schedule(self) -> SchedulerOutputs:
        # Blocks that need to be swaped or copied before model execution.
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}

        # Fix the current time.
        now = time.monotonic()

        # Join waiting sequences if possible.
        if not self.swapped:
            ignored_seq_groups: List[SequenceGroup] = []
            scheduled: List[SequenceGroup] = []
            # The total number of sequences on the fly, including the
            # requests in the generation phase.
            if self.tcp_client is not None:
                num_curr_seqs = len(self.running)
            else:
                num_curr_seqs = sum(seq_group.get_max_num_running_seqs()
                                    for seq_group in self.running)
            num_batched_tokens = 0
            # Optimization: We do not sort the waiting queue since the preempted
            # sequence groups are added to the front and the new sequence groups
            # are added to the back.
            batch_status = None
            while self.waiting:
                seq_group = self.waiting[0]

                assert seq_group.num_seqs() == 1, (
                    "Waiting sequence group should have only one prompt "
                    "sequence.")
                num_prompt_tokens = seq_group.get_seqs()[0].get_len()
                if num_prompt_tokens > self.prompt_limit:
                    logger.warning(
                        f"Input prompt ({num_prompt_tokens} tokens) is too long"
                        f" and exceeds limit of {self.prompt_limit}")
                    for seq in seq_group.get_seqs():
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored_seq_groups.append(seq_group)
                    self.waiting.pop(0)
                    continue

                # If the number of batched tokens exceeds the limit, stop.
                if (num_batched_tokens + num_prompt_tokens >
                        self.scheduler_config.max_num_batched_tokens):
                    batch_status = "TOKENFULL"
                    break

                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                num_new_seqs = seq_group.get_max_num_running_seqs()
                if (num_curr_seqs + num_new_seqs >
                        self.scheduler_config.max_num_seqs):
                    batch_status = "REQFULL"
                    break

                # If the sequence group cannot be allocated, stop.
                if self.tcp_client is None:
                    if not self.block_manager.can_allocate(seq_group):
                        break
                    ret_status = self._allocate(seq_group)
                else:
                    ret_status = self._allocate(seq_group, layerwise=True)
                    if ret_status < 0:
                        batch_status = "NOCACHE"
                        break

                if ret_status == 1:
                    seq_group = self.waiting.pop(0)
                    self.running.append(seq_group)
                    num_curr_seqs += num_new_seqs
                    continue

                seq_group = self.waiting.pop(0)
                self.running.append(seq_group)
                num_batched_tokens += num_prompt_tokens
                num_curr_seqs += num_new_seqs
                scheduled.append(seq_group)

            if self.waiting and self.tcp_client is not None:
                request_ids = [seq_group.request_id for seq_group in self.waiting]
                dump_to_shared_var(self.preempt_req_shm_name, request_ids)
                self.release_seq_group(request_ids)
                self.waiting = []
                logger.info(f"Preempte {request_ids} due to {batch_status} in prefill")

                sched_ids = [seq_group.request_id for seq_group in scheduled]

            if scheduled or ignored_seq_groups:
                scheduler_outputs = SchedulerOutputs(
                    scheduled_seq_groups=scheduled,
                    prompt_run=True,
                    num_batched_tokens=num_batched_tokens,
                    blocks_to_swap_in=blocks_to_swap_in,
                    blocks_to_swap_out=blocks_to_swap_out,
                    blocks_to_copy=blocks_to_copy,
                    ignored_seq_groups=ignored_seq_groups,
                )
                return scheduler_outputs

        # NOTE(woosuk): Preemption happens only when there is no available slot
        # to keep all the sequence groups in the RUNNING state.
        # In this case, the policy is responsible for deciding which sequence
        # groups to preempt.
        self.running = self.policy.sort_by_priority(now, self.running)

        # FIXME(MuxServe): we disable swapping for now.
        # Reserve new token slots for the running sequence groups.
        running: List[SequenceGroup] = []
        preempted: List[SequenceGroup] = []
        while self.running:
            seq_group = self.running.pop(0)

            # MuxServe schedule
            if self.tcp_client is not None:
                while self._append_slot(seq_group, None, layerwise=True) < 0:
                    if self.running:
                        # Preempt the lowest-priority sequence groups.
                        victim_seq_group = self.running.pop(-1)
                        self._preempt(victim_seq_group, blocks_to_swap_out)
                        preempted.append(victim_seq_group)
                    else:
                        # No other sequence groups can be preempted.
                        # Preempt the current sequence group.
                        self._preempt(seq_group, blocks_to_swap_out)
                        preempted.append(seq_group)
                        break
                else:
                    running.append(seq_group)
                continue

            while not self.block_manager.can_append_slot(seq_group):
                if self.running:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq_group = self.running.pop(-1)
                    self._preempt(victim_seq_group, blocks_to_swap_out)
                    preempted.append(victim_seq_group)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    self._preempt(seq_group, blocks_to_swap_out)
                    preempted.append(seq_group)
                    break
            else:
                # Append new slots to the sequence group.
                self._append_slot(seq_group, blocks_to_copy)
                running.append(seq_group)
        self.executing.extend(running)

        # Swap in the sequence groups in the SWAPPED state if possible.
        self.swapped = self.policy.sort_by_priority(now, self.swapped)
        if not preempted:
            if self.tcp_client is not None:
                num_curr_seqs = len(self.running)
            else:
                num_curr_seqs = sum(seq_group.get_max_num_running_seqs()
                                for seq_group in running)

            while self.swapped:
                seq_group = self.swapped[0]
                # If the sequence group cannot be swapped in, stop.
                if not self.block_manager.can_swap_in(seq_group):
                    break

                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                num_new_seqs = seq_group.get_max_num_running_seqs()
                if (num_curr_seqs + num_new_seqs >
                        self.scheduler_config.max_num_seqs):
                    break

                seq_group = self.swapped.pop(0)
                self._swap_in(seq_group, blocks_to_swap_in)
                self._append_slot(seq_group, blocks_to_copy)
                num_curr_seqs += num_new_seqs
                running.append(seq_group)

        if self.tcp_client is not None and preempted:
            request_ids = [seq_group.request_id for seq_group in preempted]
            dump_to_shared_var(self.preempt_req_shm_name, request_ids)
            logger.info(f"Preempte {request_ids} due to NOCACHE in decoding")

        # Each sequence in the generation phase only takes one token slot.
        # Therefore, the number of batched tokens is equal to the number of
        # sequences in the RUNNING state.
        if self.tcp_client is not None:
            num_batched_tokens = len(self.running)
        else:
            num_batched_tokens = sum(
                seq_group.num_seqs(status=SequenceStatus.RUNNING)
                for seq_group in running
            )

        if self.tcp_client is None:
            self.running = running
            self.executing = []

        scheduler_outputs = SchedulerOutputs(
            scheduled_seq_groups=running,
            prompt_run=False,
            num_batched_tokens=num_batched_tokens,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=[],
        )
        return scheduler_outputs

    def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
        scheduler_outputs = self._schedule()

        # Create input data structures.
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for seq_group in scheduler_outputs.scheduled_seq_groups:
            seq_data: Dict[int, List[SequenceData]] = {}
            block_tables: Dict[int, List[int]] = {}
            layer_block_tables: Dict[int, List[List[int]]] = {}
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                if self.tcp_client is None:
                    block_tables[seq_id] = self.block_manager.get_block_table(seq)
                else:
                    layer_block_tables[seq_id] = self.block_manager.get_layer_block_table(seq)

            seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group.request_id,
                is_prompt=scheduler_outputs.prompt_run,
                seq_data=seq_data,
                sampling_params=seq_group.sampling_params,
                block_tables=block_tables,
                layer_block_tables=layer_block_tables,
                is_muxserve=self.tcp_client is not None,
            )
            seq_group_metadata_list.append(seq_group_metadata)
        return seq_group_metadata_list, scheduler_outputs

    def fork_seq(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        self.block_manager.fork(parent_seq, child_seq)

    def free_seq(self, seq: Sequence) -> None:
        self.block_manager.free(seq)

    def free_finished_seq_groups(self) -> None:
        self.running = [
            seq_group for seq_group in self.running
            if not seq_group.is_finished()
        ]

    def _allocate(self, seq_group: SequenceGroup, layerwise=False) -> None:
        if layerwise:
            ret_status = self.block_manager.layerwise_allocate(seq_group)
            if ret_status < 0:
                return ret_status
        else:
            self.block_manager.allocate(seq_group)
            ret_status = 0
        for seq in seq_group.get_seqs():
            seq.status = SequenceStatus.RUNNING
        return ret_status

    def _append_slot(
        self,
        seq_group: SequenceGroup,
        blocks_to_copy: Optional[Dict[int, List[int]]] = None,
        layerwise: bool = False,
    ) -> None:
        assert False
        if layerwise:
            # we assume only on seq in one seqgroup since we disable beam search
            for seq in seq_group.get_seqs():
                ret = self.block_manager.layerwise_append_slot(seq)
                return ret
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            ret = self.block_manager.append_slot(seq)
            if ret is not None:
                src_block, dst_block = ret
                if src_block in blocks_to_copy:
                    blocks_to_copy[src_block].append(dst_block)
                else:
                    blocks_to_copy[src_block] = [dst_block]

    def _preempt(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
        preemption_mode: Optional[PreemptionMode] = None,
    ) -> None:
        # If preemption mode is not specified, we determine the mode as follows:
        # We use recomputation by default since it incurs lower overhead than
        # swapping. However, when the sequence group has multiple sequences
        # (e.g., beam search), recomputation is not currently supported. In
        # such a case, we use swapping instead.
        # FIXME(woosuk): This makes our scheduling policy a bit bizarre.
        # As swapped sequences are prioritized over waiting sequences,
        # sequence groups with multiple sequences are implicitly prioritized
        # over sequence groups with a single sequence.
        # TODO(woosuk): Support recomputation for sequence groups with multiple
        # sequences. This may require a more sophisticated CUDA kernel.
        if preemption_mode is None:
            if self.tcp_client is not None:
                preemption_mode = PreemptionMode.ABORT
            elif seq_group.get_max_num_running_seqs() == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP
        if preemption_mode == PreemptionMode.ABORT:
            self._preempt_by_abort(seq_group)
        elif preemption_mode == PreemptionMode.RECOMPUTE:
            self._preempt_by_recompute(seq_group)
        elif preemption_mode == PreemptionMode.SWAP:
            self._preempt_by_swap(seq_group, blocks_to_swap_out)
        else:
            assert False, "Invalid preemption mode."

    def _preempt_by_abort(self, seq_group: SequenceGroup) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        assert len(seqs) == 1
        for seq in seqs:
            seq.status = SequenceStatus.FINISHED_ABORTED
            self.block_manager.free(seq)
            self.requests_in_queue.remove(seq.seq_id)

    def _preempt_by_recompute(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        assert len(seqs) == 1
        for seq in seqs:
            seq.status = SequenceStatus.WAITING
            self.block_manager.free(seq)
        # NOTE: For FCFS, we insert the preempted sequence group to the front
        # of the waiting queue.
        self.waiting.insert(0, seq_group)

    def _preempt_by_swap(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        self._swap_out(seq_group, blocks_to_swap_out)
        self.swapped.append(seq_group)

    def _swap_in(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_in: Dict[int, int],
    ) -> None:
        mapping = self.block_manager.swap_in(seq_group)
        blocks_to_swap_in.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            seq.status = SequenceStatus.RUNNING

    def _swap_out(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        if not self.block_manager.can_swap_out(seq_group):
            # FIXME(woosuk): Abort the sequence group instead of aborting the
            # entire engine.
            raise RuntimeError(
                "Aborted due to the lack of CPU swap space. Please increase "
                "the swap space to avoid this error.")
        mapping = self.block_manager.swap_out(seq_group)
        blocks_to_swap_out.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = SequenceStatus.SWAPPED
