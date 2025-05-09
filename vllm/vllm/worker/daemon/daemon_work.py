from vllm.distributed import *
from vllm.attention.backends.flash_attn import *
from .mini_allocator import MiniKVManager
import torch
from vllm.distributed.parallel_state import get_attn_meta
from vllm.distributed.device_communicators.pynccl_wrapper import (
    nccl_group_start,
    nccl_group_end,
)
chunk_enabled = os.getenv("CHUNKE_ENABLED", "1") == "1"
abal_use_v0 = os.getenv("ABAL_USE_V0", "0") == "1"
print(f"""==================================
Daemon worker is using v0: {abal_use_v0}
Daemon worker is using chunk enabled: {chunk_enabled}
==================================""")
print(f"""==================================
Daemon worker is using v0: {abal_use_v0}
Daemon worker is using chunk enabled: {chunk_enabled}
==================================""")
print(f"""==================================
Daemon worker is using v0: {abal_use_v0}
Daemon worker is using chunk enabled: {chunk_enabled}
==================================""")
embedding_splits = 8


def do_attention(
    query, key, value, key_cache, value_cache, slot_mapping, seq_lens, block_tables, check_meta
):
    window_size = [-1, -1]
    logits_soft_cap = 0.0
    alibi_slopes = None
    softmax_scale = 128**-0.5
    # torch.ops._C_cache_ops.reshape_and_cache_flash(
    #     key,
    #     value,
    #     key_cache,
    #     value_cache,
    #     slot_mapping.flatten(),  # type: ignore[union-attr]
    #     "auto",
    #     1.0,
    #     1.0,
    # )
    return flash_attn_with_kvcache(
        q=query.unsqueeze(1),
        k_cache=key_cache,
        v_cache=value_cache,
        check_meta = get_attn_meta()[1],
        block_table=block_tables,
        cache_seqlens=seq_lens,
        softmax_scale=softmax_scale,
        causal=True,
        softcap=logits_soft_cap,
    ).squeeze(1)

DAEMON_WORKER = None

class DaemonWorkerV0:
    def __init__(self, vllm_kv_allocator, vllm_kv_caches):
        assert DAEMON_WORKER is None
        self.manager = MiniKVManager(vllm_kv_allocator, vllm_kv_caches)
        self.vllm_kv_caches = vllm_kv_caches
        # receive the sequence group meta
        bsz = 256
        self.bsz = bsz
        self.block_tables = torch.zeros(bsz, 256, dtype=torch.int32, device="cuda")
        self.block_tables_shadow = torch.zeros(
            bsz, 256, dtype=torch.int32, device="cpu", pin_memory=True
        )
        self.seq_lens = torch.zeros(bsz, dtype=torch.int32, device="cuda")
        self.seq_lens_shadow = torch.zeros(
            bsz, dtype=torch.int32, device="cpu", pin_memory=True
        )
        self.slot_mapping = torch.zeros(bsz, 1, dtype=torch.int64, device="cuda")
        self.slot_mapping_shadow = torch.zeros(
            bsz, 1, dtype=torch.int64, device="cpu", pin_memory=True
        )
        self.query_key_value = torch.zeros(
            bsz, 32 + 8 + 8, 128, dtype=torch.bfloat16, device="cuda"
        )
       
        self.remaining_step = 0
        self.total_tokens = 0
        self.transmission_stream = torch.cuda.Stream()
        self.graphs = {}
        self.warmup()
        
        torch.distributed.barrier()
        self.out = torch.zeros(256, 32, 128, dtype=torch.bfloat16, device="cuda")
    def warmup(self):
        for seq in range(150):
            self.manager.allocate_kv_cache(seq, 70)
        self.seq_lens.fill_(70*16)
        self.slot_mapping.fill_(70*16-1)
        self.block_tables.fill_(0)
        for seq in range(1,150):
            self.capture(seq)
        for seq in range(150):
            self.manager.free(seq)
    def capture(self, total_tokens):
        if total_tokens in self.graphs:
            return
        
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            layer = 0
            query = self.query_key_value[:, :32]
            key = self.query_key_value[:, 32 : 32 + 8]
            value = self.query_key_value[:, 32 +8 :]
           
            out = do_attention(
                query[0:total_tokens],
                key[0:total_tokens],
                value[0:total_tokens],
                self.vllm_kv_caches[layer][0],
                self.vllm_kv_caches[layer][1],
                self.slot_mapping[0:total_tokens],
                self.seq_lens[0:total_tokens],
                self.block_tables[0:total_tokens],
                check_meta=None
            )
            
        self.graphs[total_tokens] = graph
    def step(self):
        if not self.transmission_stream.query():
            return
        if self.remaining_step > 0:
            # check if transmission stream is over
            layer = 32 - self.remaining_step
            if self.total_tokens not in self.graphs:
                self.capture(self.total_tokens)
            self.graphs[self.total_tokens].replay()
            # print("send tensor to the next layer, finish ", self.total_tokens, self.remaining_step,flush=True)
            sr_tensor(self.out[:self.total_tokens], 1, 0)
            # print("send tensor to the next layer, finish ", self.total_tokens, self.remaining_step,flush=True)
            
            self.remaining_step -= 1
            if not self.remaining_step == 0:
                sr_tensor(
                    self.query_key_value[0:self.total_tokens],
                    0,
                    1,
                    stream=self.transmission_stream,
                )
            else:
                pass
        else:
            req = sr_obj_async(None, 0, 1)
            if req is None:
                return
            if req["type"] == "migrate":
                for idx in range(req["seq_nums"]):
                    self.manager.allocate_kv_cache(
                        req["seq_ids"][idx], req["seq_lengths"][idx]
                    )
                    physical_block_ids = self.manager.block_tables[
                        req["seq_ids"][idx]
                    ].physical_block_ids
                    num_blocks = len(physical_block_ids)
                    

            elif req["type"] == "meta":
                # get the number of tokens to recv from the sender
                # print(req)
                total_tokens = req["seq_count"]
                # print(total_tokens)
                for idx in range(req["seq_count"]):
                    self.manager.allocate_kv_cache(
                        req["seq_ids"][idx], req["seq_lengths"][idx]
                    )
                for idx in range(req["seq_count"]):
                    physical_block_ids = self.manager.block_tables[
                        req["seq_ids"][idx]
                    ].physical_block_ids
                    num_blocks = len(physical_block_ids)
                    self.block_tables_shadow[idx][0:num_blocks] = torch.tensor(
                        physical_block_ids, dtype=torch.int32
                    )
                    self.seq_lens_shadow[idx] = req["seq_lengths"][idx]
                    # print(physical_block_ids, req["seq_lengths"][idx])
                    self.slot_mapping_shadow[idx] = (
                        req["seq_lengths"][idx] - 1
                    ) % 16 + physical_block_ids[-1] * 16
                self.block_tables.copy_(self.block_tables_shadow, non_blocking=True)
                self.seq_lens.copy_(self.seq_lens_shadow, non_blocking=True)
                self.slot_mapping.copy_(self.slot_mapping_shadow, non_blocking=True)
                self.remaining_step = 256
                self.total_tokens = total_tokens
                sr_tensor(
                    self.query_key_value[0:total_tokens],
                    0,
                    1,
                    stream=self.transmission_stream,
                )
            elif req["type"] == "free":
                for idx in req["seq_ids"]:
                    print("freeing", idx)
                    self.manager.free(idx)
class DaemonWorker:
    def __init__(self, vllm_kv_allocator, vllm_kv_caches):
        assert DAEMON_WORKER is None
        self.manager = MiniKVManager(vllm_kv_allocator, vllm_kv_caches)
        self.vllm_kv_caches = vllm_kv_caches
        # receive the sequence group meta
        bsz = 200
        self.bsz = bsz
        self.block_tables = torch.zeros(bsz, 256, dtype=torch.int32, device="cuda")
        self.block_tables_shadow = torch.zeros(
            bsz, 256, dtype=torch.int32, device="cpu", pin_memory=True
        )
        self.seq_lens = torch.zeros(bsz, dtype=torch.int32, device="cuda")
        self.seq_lens_shadow = torch.zeros(
            bsz, dtype=torch.int32, device="cpu", pin_memory=True
        )
        self.slot_mapping = torch.zeros(bsz, 1, dtype=torch.int64, device="cuda")
        self.slot_mapping_shadow = torch.zeros(
            bsz, 1, dtype=torch.int64, device="cpu", pin_memory=True
        )
        self.query_key_value = get_attn_meta()[2].view(200, 32 + 8 + 8, 128)
        
        self.remaining_step = 0
        self.total_tokens = 0
        self.transmission_stream = torch.cuda.Stream()
        self.check_meta = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.graphs = {}
        self.total_tokens = 1
        self.remaining_step = 32
        self.warmup()
        self.total_tokens = 1
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        torch.distributed.barrier()
        print("daemon warm up first")
        
    def warmup(self):
        for seq in range(192):
            self.manager.allocate_kv_cache(seq, 70)
        self.seq_lens.fill_(70*16)
        self.slot_mapping.fill_(70*16-1)
        self.block_tables.fill_(0)
        for seq in range(1,192):
            self.capture(seq)
        for seq in range(192):
            self.manager.free(seq)
            
    def capture(self, total_tokens):
        if total_tokens in self.graphs:
            return
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            query = self.query_key_value[:, :32]
            key = self.query_key_value[:, 32 : 32 + 8]
            value = self.query_key_value[:, 32 + 8 :]
            layer = 0
            
            out = do_attention(
                query[0:total_tokens],
                key[0:total_tokens],
                value[0:total_tokens],
                self.vllm_kv_caches[layer][0],
                self.vllm_kv_caches[layer][1],
                self.slot_mapping[0:total_tokens],
                self.seq_lens[0:total_tokens],
                self.block_tables[0:total_tokens],
                self.check_meta
            )
            # dst = self.query_key_value.view(-1, 4096)
            # dst[0:total_tokens].copy_(out.view(-1, 4096), non_blocking=True)
            torch.ops._C.conditional_copy(get_attn_meta()[1], out, self.query_key_value, total_tokens)
            torch.ops._C.update_offload(get_attn_meta()[1])
        self.graphs[total_tokens] = graph
    def step(self):
        # if not self.transmission_stream.query():
        #     return
        if self.remaining_step > 0:
            self.graphs[self.total_tokens].replay()
            
        req = sr_obj_async(None, 0, 1)
        if req is None:
            return
        if req["type"] == "migrate":
            for idx in range(req["seq_nums"]):
                self.manager.allocate_kv_cache(
                    req["seq_ids"][idx], req["seq_lengths"][idx]
                )
                physical_block_ids = self.manager.block_tables[
                    req["seq_ids"][idx]
                ].physical_block_ids
                num_blocks = len(physical_block_ids)
                # for layer in range(32):
                #     sr_tensor(self.kv_cache_tmp[0, :num_blocks], 0, 1)
                #     sr_tensor(self.kv_cache_tmp[1, :num_blocks], 0, 1)
                #     self.manager.kv_cache[layer][:, physical_block_ids] = (
                #         self.kv_cache_tmp[:, :num_blocks]
                #     )
            # receive the migrate

        elif req["type"] == "meta":
            # get the number of tokens to recv from the sender
            # print(req)
            total_tokens = req["seq_count"]
            for idx in range(req["seq_count"]):
                self.manager.allocate_kv_cache(
                    req["seq_ids"][idx], req["seq_lengths"][idx]
                )
            for idx in range(req["seq_count"]):
                physical_block_ids = self.manager.block_tables[
                    req["seq_ids"][idx]
                ].physical_block_ids
                num_blocks = len(physical_block_ids)
                self.block_tables_shadow[idx][0:num_blocks] = torch.tensor(
                    physical_block_ids, dtype=torch.int32
                )
                self.seq_lens_shadow[idx] = req["seq_lengths"][idx]
                # print(physical_block_ids, req["seq_lengths"][idx])
                self.slot_mapping_shadow[idx] = (
                    req["seq_lengths"][idx] - 1
                ) % 16 + physical_block_ids[-1] * 16
            self.block_tables.copy_(self.block_tables_shadow, non_blocking=True)
            self.seq_lens.copy_(self.seq_lens_shadow, non_blocking=True)
            self.slot_mapping.copy_(self.slot_mapping_shadow, non_blocking=True)

            self.remaining_step = 32
            self.total_tokens = total_tokens
            if self.total_tokens in self.graphs:
                pass
            else:
                self.capture(total_tokens)
        elif req["type"] == "free":
            for idx in req["seq_ids"]:
                print("freeing", idx)
                self.manager.free(idx)
        else:
            assert False


def init_daemon_worker(vllm_kv_allocator, vllm_kv_caches):
    global DAEMON_WORKER
    if abal_use_v0:
        DAEMON_WORKER = DaemonWorkerV0(vllm_kv_allocator, vllm_kv_caches)
    else:
        DAEMON_WORKER = DaemonWorker(vllm_kv_allocator, vllm_kv_caches)


def print_daemon_worker():
    print(DAEMON_WORKER)

step_count = 0
def step():
    global step_count
    step_count += 1
    if get_role() == "RECEIVER" and offload_enabled():
        DAEMON_WORKER.step()
def print_step_count():
    global step_count
    print("the step count is ", step_count)
    step_count = 0

# def run_daemon_loop(vllm_kv_allocator, vllm_kv_caches):

#     while True:
#             # print(slot_mapping)
#             # print(slot_mapping_shadow)

#             # we have to do some kvcache pagetable and slotmapping things here!
#             for layer in range(32):
#                 # print(layer)
#                 # we need to receive the query, key, value from the sender
#                 # nccl_group_start()
#                 # "sr_tensor"(query[0:total_tokens], 0, 1)
#                 # sr_tensor(key[0:total_tokens], 0, 1)
#                 # sr_tensor(value[0:total_tokens], 0, 1)
#                 sr_tensor(query_key_value[0:total_tokens], 0, 1)
#                 query = query_key_value[:, :32]
#                 key = query_key_value[:, 32 : 32 + 8]
#                 value = query_key_value[:, 32 + 8 :]

#                 # nccl_group_end()
#                 # print(query[0:total_tokens].mean())
#                 # print(key[0:total_tokens].mean())
#                 # print(value[0:total_tokens].mean())

#                 out = do_attention(
#                     query[0:total_tokens],
#                     key[0:total_tokens],
#                     value[0:total_tokens],
#                     vllm_kv_caches[layer][0],
#                     vllm_kv_caches[layer][1],
#                     slot_mapping[0:total_tokens],
#                     seq_lens[0:total_tokens],
#                     block_tables[0:total_tokens],
#                 )
#                 # print(out.mean())
#                 sr_tensor(out, 1, 0)
#             # receive the meta
#         else:
#             assert False
#         pass


if __name__ == "__main__":
    run_daemon_loop()
