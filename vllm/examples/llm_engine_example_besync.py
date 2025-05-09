import argparse
from typing import List, Tuple

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.utils import FlexibleArgumentParser

import time, torch
import json
import random
import numpy as np

def create_test_prompts(args) -> List[Tuple[str, SamplingParams]]:
    """Create a list of test prompts with their sampling parameters."""
    # torch.cuda.set_sync_debug_mode("warn")
    #
    if False:
        return [
            (
                "To " * 1000,
                SamplingParams(temperature=0.0, max_tokens=100, ignore_eos=True),
            )
            for _ in range(1200)
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
    else:
        with open(args.dataset) as f:
            dataset = json.loads(f.read())
            random.seed(20250106)
            np.random.seed(20250106)
            sampled_requests = random.sample(dataset, 10000)
            all_res = []
            for req in sampled_requests:
                input_len = req[0]
                output_len = req[1]
                # output_len = min(128,output_len)
                all_res.append((
                    "To " * input_len,
                    SamplingParams(temperature=0.0, max_tokens=output_len, ignore_eos=True)),
                )
            return all_res
                    
        
def process_requests(engine: LLMEngine, test_prompts: List[Tuple[str, SamplingParams]],args):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0
    # first we add all the prompts to the engine

    prompt, sampling_params = test_prompts.pop(0)
    last_time = time.time()

    engine.add_request(str(request_id), prompt, sampling_params)
    request_id += 1
    last_time = time.time()

    start_time = time.time()
    hit_count = 0
    while True:
        
        request_outputs: List[RequestOutput] = engine.step()
        if time.time() - last_time > 1/args.qps:
            prompt, sampling_params = test_prompts.pop(0)
            # hit_count += 1
            # print("the hit count is ", hit_count)
            # if hit_count == 12:
            #     exit()
            print(request_id)
            engine.add_request(str(request_id), prompt, sampling_params)
            request_id += 1
            last_time = time.time()
        # for request_output in request_outputs:
        #     if request_output.finished:
        #         print(request_output.request_id, request_output.outputs[0].text[:50])


def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine(args)
    test_prompts = create_test_prompts(args)
    process_requests(engine, test_prompts,args)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Demo on using the LLMEngine class directly"
    )
    parser.add_argument("--dataset", type=str, default= "~/weaver/burst.json", help="Path to the dataset")
    parser.add_argument("--qps", type=float, default=1, help="QPS")
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
