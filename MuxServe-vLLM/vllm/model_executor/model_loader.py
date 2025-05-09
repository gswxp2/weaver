"""Utilities for selecting and loading models."""
import contextlib
from typing import Type, List, Optional, Dict

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config import ModelConfig
from vllm.model_executor.models import *  # pylint: disable=wildcard-import
from vllm.model_executor.weight_utils import (get_quant_config,
                                              initialize_dummy_weights)
# for muxserve
from torch.multiprocessing.reductions import rebuild_cuda_tensor
from vllm.zmq_tool import ZMQClient

from vllm.logger import init_logger
logger = init_logger(__name__)

# TODO(woosuk): Lazy-load the model classes.
_MODEL_REGISTRY = {
    "AquilaModel": AquilaForCausalLM,
    "BaiChuanForCausalLM": BaiChuanForCausalLM,  # baichuan-7b
    "BaichuanForCausalLM": BaichuanForCausalLM,  # baichuan-13b
    "BloomForCausalLM": BloomForCausalLM,
    "FalconForCausalLM": FalconForCausalLM,
    "GPT2LMHeadModel": GPT2LMHeadModel,
    "GPTBigCodeForCausalLM": GPTBigCodeForCausalLM,
    "GPTJForCausalLM": GPTJForCausalLM,
    "GPTNeoXForCausalLM": GPTNeoXForCausalLM,
    "InternLMForCausalLM": InternLMForCausalLM,
    "LlamaForCausalLM": LlamaForCausalLM,
    "LLaMAForCausalLM": LlamaForCausalLM,  # For decapoda-research/llama-*
    "MistralForCausalLM": MistralForCausalLM,
    "MPTForCausalLM": MPTForCausalLM,
    "OPTForCausalLM": OPTForCausalLM,
    "QWenLMHeadModel": QWenLMHeadModel,
    "RWForCausalLM": FalconForCausalLM,
}

# FIXME(woosuk): Remove this once all models support quantization.
_MODEL_CLASSES_SUPPORT_QUANTIZATION = [
    LlamaForCausalLM,
    MistralForCausalLM,
]


@contextlib.contextmanager
def _set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def _get_model_architecture(config: PretrainedConfig) -> Type[nn.Module]:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _MODEL_REGISTRY:
            return _MODEL_REGISTRY[arch]
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {list(_MODEL_REGISTRY.keys())}")


'''
assume the format of data the client recieves from the server like:
data: Dict[str, dict]

{
    "model.embed_tokens.weight": {"tensor_size": [],
            "tensor_stride": [],
            "tensor_offset": [],
            "storage_cls": [],
            "dtype": [],
            "storage_device": [],
            "storage_handle": [],
            "storage_size_bytes": [],
            "storage_offset_bytes": [],
            "requires_grad": [],
            "ref_counter_handle": [],
            "ref_counter_offset": [],
            "event_handle": [],
            "event_sync_required": [],},

    "model.layers.0.input_layernorm.weight": {"tensor_size": [],
            "tensor_stride": [],
            "tensor_offset": [],
            "storage_cls": [],
            "dtype": [],
            "storage_device": [],
            "storage_handle": [],
            "storage_size_bytes": [],
            "storage_offset_bytes": [],
            "requires_grad": [],
            "ref_counter_handle": [],
            "ref_counter_offset": [],
            "event_handle": [],
            "event_sync_required": [],},

    "model.layers.0.post_attention_layernorm.weight": {
        ...
    },

    "param_name": {
        handler for rebuild the tensor
    },

    ...

}
'''

def update_parameters(model:nn.Module, data: Dict[str, dict]):
    for param_name, param in model.named_parameters():
        cuda_tensor = rebuild_cuda_tensor(torch.Tensor, **(data[param_name]))
        assert param.shape == cuda_tensor.shape
        assert cuda_tensor.is_cuda
        param.data = cuda_tensor

# add the parallel load version for memo :)
# import concurrent.futures
# def update_parameters_parallel(model: nn.Module, data: Dict[str, dict]):
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         futures = {executor.submit(rebuild_cuda_tensor, torch.Tensor, **(data[param_name])): (param_name, param) for param_name, param in model.named_parameters()}

#     for future in concurrent.futures.as_completed(futures):
#         param_name, param = futures[future]
#         cuda_tensor = future.result()
#         assert param.shape == cuda_tensor.shape
#         assert cuda_tensor.is_cuda
#         param.data = cuda_tensor

def load_from_server(model:nn.Module, tcp_client: ZMQClient, model_config: ModelConfig):
    # suppose our model was deployed on single card now
    logger.info('connecting server' f'from client cuda{str(torch.cuda.current_device())}')

    # ask for the server about the weight
    rank = torch.distributed.get_rank()
    tcp_client.send_pyobj(
        ["weight", [rank, model_config.model]]
    )

    # [TODO](@runyu)most operation of this func could be asnyc operated.
    logger.info('connected, waiting data' 'client')
    data = tcp_client.recv_pyobj()
    logger.info('data received, rebuilding and printing' 'client')

    update_parameters(model, data)
    model = model.cuda() # could be commented because of assert cuda_tensor.is_cuda
    # logger.info(f'Model {model_config.model} has been initialized, update consumes time {t2 - t1}, update_parallel consumes time {t4 - t2}, model.cuda() consumes time {t3 - t2}')

def get_model(model_config: ModelConfig, tcp_client: Optional[ZMQClient]=None) -> nn.Module:
    model_class = _get_model_architecture(model_config.hf_config)

    # Get the quantization config.
    quant_config = None
    if model_config.quantization is not None:
        if model_class not in _MODEL_CLASSES_SUPPORT_QUANTIZATION:
            raise ValueError(
                f"Quantization is not supported for {model_class}.")
        quant_config = get_quant_config(model_config.quantization,
                                        model_config.model,
                                        model_config.download_dir)
        capability = torch.cuda.get_device_capability()
        capability = capability[0] * 10 + capability[1]
        if capability < quant_config.get_min_capability():
            raise ValueError(
                f"The quantization method {model_config.quantization} is not "
                "supported for the current GPU. "
                f"Minimum capability: {quant_config.get_min_capability()}. "
                f"Current capability: {capability}.")
        supported_dtypes = quant_config.get_supported_act_dtypes()
        if model_config.dtype not in supported_dtypes:
            raise ValueError(
                f"{model_config.dtype} is not supported for quantization "
                f"method {model_config.quantization}. Supported dtypes: "
                f"{supported_dtypes}")

    with _set_default_torch_dtype(model_config.dtype):

        # Create a model instance.
        # The weights will be initialized as empty tensors.
        if model_class in _MODEL_CLASSES_SUPPORT_QUANTIZATION:
            model = model_class(model_config.hf_config, quant_config)
        else:
            model = model_class(model_config.hf_config)

        if tcp_client is not None:
            load_from_server(model, tcp_client, model_config)

        else:

            if model_config.load_format == "dummy":
                model = model.cuda()
                # NOTE(woosuk): For accurate performance evaluation, we assign
                # random values to the weights.
                initialize_dummy_weights(model)
            else:
                # Load the weights from the cached or downloaded files.
                model.load_weights(model_config.model, model_config.download_dir,
                                   model_config.load_format, model_config.revision)
                model = model.cuda()

    return model.eval()
