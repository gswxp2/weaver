import os
import asyncio
import time
from functools import partial
from typing import (Any, Dict, Iterable, List, Optional, Set, Tuple, Type,
                    Union)

from muxserve.constants import SM_HOLD_NAME_FMT
from muxserve.shm_utils import create_shared_var, read_shared_var, write_shared_var

from vllm.config import ModelConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.engine.ray_utils import initialize_cluster, ray
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams

logger = init_logger(__name__)


class AsyncEngineDeadError(RuntimeError):
    pass


def _raise_exception_on_finish(task: asyncio.Task,
                               request_tracker: "RequestTracker") -> None:
    msg = ("Task finished unexpectedly. This should never happen! "
           "Please open an issue on Github.")
    try:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            raise AsyncEngineDeadError(
                msg + " See stack trace above for the actual cause.") from exc
        raise AsyncEngineDeadError(msg)
    except Exception as exc:
        request_tracker.propagate_exception(exc)
        raise exc


class AsyncStream:
    """A stream of RequestOutputs for a request that can be
    iterated over asynchronously."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self._queue = asyncio.Queue()
        self._finished = False

    def put(self, item: RequestOutput) -> None:
        if self._finished:
            return
        self._queue.put_nowait(item)

    def finish(self) -> None:
        self._queue.put_nowait(StopIteration)
        self._finished = True

    @property
    def finished(self) -> bool:
        return self._finished

    def __aiter__(self):
        return self

    async def __anext__(self) -> RequestOutput:
        result = await self._queue.get()
        if result is StopIteration:
            raise StopAsyncIteration
        elif isinstance(result, Exception):
            raise result
        return result


class RequestTracker:
    """Synchronous abstraction for tracking requests."""

    def __init__(self) -> None:
        self._request_streams: Dict[str, AsyncStream] = {}
        self._finished_requests: asyncio.Queue[str] = asyncio.Queue()
        self._new_requests: asyncio.Queue[Tuple[AsyncStream,
                                                dict]] = asyncio.Queue()
        self.new_requests_event = None

    def __contains__(self, item):
        return item in self._request_streams

    def init_event(self):
        self.new_requests_event = asyncio.Event()

    def propagate_exception(self,
                            exc: Exception,
                            request_id: Optional[str] = None) -> None:
        """Propagate an exception to request streams
        (all if request_id is None)."""
        if request_id is not None:
            self._request_streams[request_id].put(exc)
        else:
            for stream in self._request_streams.values():
                stream.put(exc)

    def process_request_output(self,
                               request_output: RequestOutput,
                               *,
                               verbose: bool = False) -> None:
        """Process a request output from the engine."""
        request_id = request_output.request_id

        self._request_streams[request_id].put(request_output)
        if request_output.finished:
            if verbose:
                logger.info(f"Finished request {request_id}.")
            self.abort_request(request_id)

    def add_request(self, request_id: str,
                    **engine_add_request_kwargs) -> AsyncStream:
        """Add a request to be sent to the engine on the next background
        loop iteration."""
        if request_id in self._request_streams:
            raise KeyError(f"Request {request_id} already exists.")

        stream = AsyncStream(request_id)
        self._new_requests.put_nowait((stream, {
            "request_id": request_id,
            **engine_add_request_kwargs
        }))

        self.new_requests_event.set()

        return stream

    def abort_request(self, request_id: str, *, verbose: bool = False) -> None:
        """Abort a request during next background loop iteration."""
        if verbose:
            logger.info(f"Aborted request {request_id}.")

        self._finished_requests.put_nowait(request_id)

        if request_id not in self._request_streams or self._request_streams[
                request_id].finished:
            # The request has already finished or been aborted.
            return

        self._request_streams[request_id].finish()

    def get_new_and_finished_requests(self) -> Tuple[List[dict], Set[str]]:
        """Get the new requests and finished requests to be
        sent to the engine."""
        new_requests: List[dict] = []
        finished_requests: Set[str] = set()

        while not self._finished_requests.empty():
            request_id = self._finished_requests.get_nowait()
            finished_requests.add(request_id)
            self._request_streams.pop(request_id, None)

        while not self._new_requests.empty():
            stream, new_request = self._new_requests.get_nowait()
            if stream.request_id in finished_requests:
                # The request has already been aborted.
                stream.finish()
                continue
            self._request_streams[stream.request_id] = stream
            new_requests.append(new_request)

        self.new_requests_event.clear()

        return new_requests, finished_requests

    def has_new_requests(self) -> bool:
        return self.new_requests_event.is_set()

    async def wait_for_new_requests(self):
        await self.new_requests_event.wait()


class _AsyncLLMEngine(LLMEngine):
    """Extension of LLMEngine to add async methods."""

    async def step_async(self) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.
        The workers are ran asynchronously if possible.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """
        seq_group_metadata_list, scheduler_outputs, ignored = self._schedule()
        if scheduler_outputs.is_empty():
            return ignored

        # Execute the model.
        output = await self._run_workers_async(
            "execute_model",
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
            blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
            blocks_to_copy=scheduler_outputs.blocks_to_copy,
        )

        return self._process_model_outputs(output, scheduler_outputs) + ignored

    async def _run_workers_async(
        self,
        method: str,
        *args,
        get_all_outputs: bool = False,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        all_outputs = []
        for worker in self.workers:
            if self.parallel_config.worker_use_ray:
                executor = partial(worker.execute_method.remote, method)
            else:
                executor = getattr(worker, method)

            output = executor(*args, **kwargs)
            all_outputs.append(output)

        if self.parallel_config.worker_use_ray:
            all_outputs = await asyncio.gather(*all_outputs)

        if get_all_outputs:
            return all_outputs

        # Make sure all workers have the same results.
        output = all_outputs[0]
        for other_output in all_outputs[1:]:
            assert output == other_output
        return output


class AsyncLLMEngine:
    """An asynchronous wrapper for LLMEngine.

    This class is used to wrap the LLMEngine class to make it asynchronous. It
    uses asyncio to create a background loop that keeps processing incoming
    requests. The LLMEngine is kicked by the generate method when there
    are requests in the waiting queue. The generate method yields the outputs
    from the LLMEngine to the caller.

    NOTE: For the comprehensive list of arguments, see `LLMEngine`.

    Args:
        worker_use_ray: Whether to use Ray for model workers. Required for
            distributed execution. Should be the same as
            `parallel_config.worker_use_ray`.
        engine_use_ray: Whether to make LLMEngine a Ray actor. If so, the
            async frontend will be executed in a separate process as the
            model workers.
        log_requests: Whether to log the requests.
        start_engine_loop: If True, the background task to run the engine
            will be automatically started in the generate call.
        *args, *kwargs: Arguments for LLMEngine.
    """

    _engine_class: Type[_AsyncLLMEngine] = _AsyncLLMEngine

    def __init__(self,
                 worker_use_ray: bool,
                 engine_use_ray: bool,
                 *args,
                 log_requests: bool = True,
                 max_log_len: Optional[int] = None,
                 start_engine_loop: bool = True,
                 **kwargs) -> None:
        self.worker_use_ray = worker_use_ray
        self.engine_use_ray = engine_use_ray
        self.log_requests = log_requests
        self.max_log_len = max_log_len
        self.engine = self._init_engine(*args, **kwargs)

        self.is_muxserve = self.engine.model_config.flexstore_port is not None
        self.model_name = self.engine.model_config.model
        if self.is_muxserve:
            mps_percent = os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", None)
            self.sm_hold_name = SM_HOLD_NAME_FMT.format(
                self.model_name, mps_percent,
            )
            self.sm_hold_var = create_shared_var(self.sm_hold_name)

        # warmup engine
        logger.info("Warmup engine...")
        assert False
        self.engine.warmup()

        self.background_loop = None
        # We need to keep a reference to unshielded
        # task as well to prevent it from being garbage
        # collected
        self._background_loop_unshielded = None
        self.start_engine_loop = start_engine_loop
        self._request_tracker = RequestTracker()

    @property
    def is_running(self) -> bool:
        return (self.background_loop is not None
                and not self.background_loop.done())

    def start_background_loop(self) -> None:
        """Start the background loop."""
        if self.is_running:
            raise RuntimeError("Background loop is already running.")
        self._request_tracker.init_event()

        # if self.is_muxserve:
        #     engine_loop_fn = self.muxserve_run_engine_loop
        # else:
        #     engine_loop_fn = self.run_engine_loop
        engine_loop_fn = self.run_engine_loop

        self._background_loop_unshielded = asyncio.get_event_loop(
        ).create_task(engine_loop_fn())
        self._background_loop_unshielded.add_done_callback(
            partial(_raise_exception_on_finish,
                    request_tracker=self._request_tracker))
        self.background_loop = asyncio.shield(self._background_loop_unshielded)

    def _init_engine(self, *args,
                     **kwargs) -> Union[_AsyncLLMEngine, "ray.ObjectRef"]:
        if not self.engine_use_ray:
            engine_class = self._engine_class
        elif self.worker_use_ray:
            engine_class = ray.remote(num_cpus=0)(self._engine_class).remote
        else:
            engine_class = ray.remote(num_gpus=1)(self._engine_class).remote
        return engine_class(*args, **kwargs)

    async def engine_step(self) -> bool:
        """Kick the engine to process the waiting requests.

        Returns True if there are in-progress requests."""

        new_requests, finished_requests = (
            self._request_tracker.get_new_and_finished_requests())

        for new_request in new_requests:
            # Add the request into the vLLM engine's waiting queue.
            # TODO: Maybe add add_request_batch to reduce Ray overhead
            if self.engine_use_ray:
                await self.engine.add_request.remote(**new_request)
            else:
                self.engine.add_request(**new_request)

        if finished_requests:
            await self._engine_abort(finished_requests)

        if self.engine_use_ray:
            request_outputs = await self.engine.step.remote()
        else:
            request_outputs = await self.engine.step_async()

        # Put the outputs into the corresponding streams.
        for request_output in request_outputs:
            self._request_tracker.process_request_output(
                request_output, verbose=self.log_requests)

        return len(request_outputs) > 0

    async def _engine_abort(self, request_ids: Iterable[str]):
        if self.engine_use_ray:
            await self.engine.abort_request.remote(request_ids)
        else:
            self.engine.abort_request(request_ids)

    async def muxserve_run_engine_loop(self):
        has_requests_in_progress = False
        while True:
            iters = read_shared_var(self.sm_hold_var)
            if iters <= 0:
                await asyncio.sleep(0.002)
                continue

            while iters > 0:
                if not has_requests_in_progress:
                    await asyncio.sleep(0)
                    if not self._request_tracker.has_new_requests():
                        break

                has_requests_in_progress = await self.engine_step()
                iters -= 1
                await asyncio.sleep(0)

            write_shared_var(self.sm_hold_var, 0)

    async def run_engine_loop(self):
        # Initialize the RequestTracker here so it uses the right event loop.
        has_requests_in_progress = False
        while True:
            if not has_requests_in_progress:
                await self._request_tracker.wait_for_new_requests()
            has_requests_in_progress = await self.engine_step()
            await asyncio.sleep(0)

    async def add_request(
        self,
        request_id: str,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        is_free_cache: Optional[bool] = None,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
    ) -> AsyncStream:
        if self.log_requests:
            shortened_prompt = prompt
            shortened_token_ids = prompt_token_ids
            if self.max_log_len is not None:
                if shortened_prompt is not None:
                    shortened_prompt = shortened_prompt[:self.max_log_len]
                if shortened_token_ids is not None:
                    shortened_token_ids = shortened_token_ids[:self.
                                                              max_log_len]
            logger.info(f"Received request {request_id}: "
                        f"prompt: {shortened_prompt!r}, "
                        f"is_free_cache: {is_free_cache},"
                        f"sampling params: {sampling_params}, "
                        f"prompt token ids: {shortened_token_ids}.")

        if not self.is_running:
            if self.start_engine_loop:
                self.start_background_loop()
            else:
                raise AsyncEngineDeadError(
                    "Background loop is not running. If it was running, "
                    "inspect the output to find the stacktrace of the "
                    "error that caused the background loop to stop "
                    "(AsyncEngineDeadError).")

        stream = self._request_tracker.add_request(
            request_id,
            prompt=prompt,
            sampling_params=sampling_params,
            is_free_cache=is_free_cache,
            prompt_token_ids=prompt_token_ids,
            arrival_time=arrival_time)

        return stream

    async def generate(
            self,
            prompt: Optional[str],
            sampling_params: SamplingParams,
            request_id: str,
            is_free_cache: Optional[bool] = None,
            prompt_token_ids: Optional[List[int]] = None) -> RequestOutput:
        """Generate outputs for a request.

        Generate outputs for a request. This method is a coroutine. It adds the
        request into the waiting queue of the LLMEngine and streams the outputs
        from the LLMEngine to the caller.

        Args:
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters of the request.
            request_id: The unique id of the request.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.

        Yields:
            The output `RequestOutput` objects from the LLMEngine for the
            request.
        """
        # Preprocess the request.
        # This should not be used for logging, as it is monotonic time.
        arrival_time = time.monotonic()

        try:
            stream = await self.add_request(request_id,
                                            prompt,
                                            sampling_params,
                                            is_free_cache=is_free_cache,
                                            prompt_token_ids=prompt_token_ids,
                                            arrival_time=arrival_time)

            async for request_output in stream:
                yield request_output
        except (Exception, asyncio.CancelledError) as e:
            # If there is an exception or coroutine is cancelled, abort the
            # request.
            self._abort(request_id)
            raise e

    async def abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        if not self.is_running:
            raise AsyncEngineDeadError(
                "Background loop is not running. If it was running, "
                "inspect the output to find the stacktrace of the "
                "error that caused the background loop to stop "
                "(AsyncEngineDeadError).")

        return self._abort(request_id)

    def _abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        self._request_tracker.abort_request(request_id,
                                            verbose=self.log_requests)

    async def get_model_config(self) -> ModelConfig:
        """Get the model configuration of the vLLM engine."""
        if self.engine_use_ray:
            return await self.engine.get_model_config.remote()
        else:
            return self.engine.get_model_config()

    @classmethod
    def from_engine_args(cls,
                         engine_args: AsyncEngineArgs,
                         start_engine_loop: bool = True) -> "AsyncLLMEngine":
        """Creates an async LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_configs = engine_args.create_engine_configs()
        parallel_config = engine_configs[2]
        ray_address = parallel_config.ray_address
        # Initialize the cluster.
        distributed_init_method, placement_group = initialize_cluster(
            parallel_config, engine_args.engine_use_ray, ray_address=ray_address)
        # Create the async LLM engine.
        engine = cls(engine_args.worker_use_ray,
                     engine_args.engine_use_ray,
                     *engine_configs,
                     distributed_init_method,
                     placement_group,
                     log_requests=not engine_args.disable_log_requests,
                     log_stats=not engine_args.disable_log_stats,
                     max_log_len=engine_args.max_log_len,
                     start_engine_loop=start_engine_loop)
        return engine
