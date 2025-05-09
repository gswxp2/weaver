import argparse
from typing import List, Tuple

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.utils import FlexibleArgumentParser

import time, torch

def create_test_prompts(args) -> List[Tuple[str, SamplingParams]]:
    """Create a list of test prompts with their sampling parameters."""
    torch.cuda.set_sync_debug_mode("warn")
    return [
        (
            "To " * 1000,
            SamplingParams(temperature=0.0, max_tokens=110, ignore_eos=True),
        )
        for _ in range(args.bs)
        # (
        #     "To be or not to be, that is a question, I have my own idea: " ,
        #     SamplingParams(temperature=0.0, max_tokens=200,ignore_eos=True ),
        # )
        # for _ in range(256)
        # (
        #     "To be or not to be, It is a problem tought by Hamlet. But I have my own answer:",
        #     SamplingParams(temperature=0.0, max_tokens=50),
        # )
        # for _ in range(30)
    ]


def process_requests(engine: LLMEngine,
                     test_prompts: List[Tuple[str, SamplingParams]]):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0
    # first we add all the prompts to the engine
    while test_prompts:
        prompt, sampling_params = test_prompts.pop(0)
        engine.add_request(str(request_id), prompt, sampling_params)
        request_id += 1
    start_time = time.time()
    while test_prompts or engine.has_unfinished_requests():
        request_outputs: List[RequestOutput] = engine.step()
        # for request_output in request_outputs:
        #     if request_output.finished:
        #         print(request_output.request_id, request_output.outputs[0].text[:50])
    end_time = time.time()
    # print(end_time - start_time)
    # print(end_time - engine.last_prefill)
    # print(end_time - engine.last_offload_time)


def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine(args)
    test_prompts = create_test_prompts(args)
    process_requests(engine, test_prompts)


if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    parser.add_argument('--bs', type=int, default=1, help='batch size')
    args = parser.parse_args()
    main(args)
