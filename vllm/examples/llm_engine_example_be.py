import argparse
from typing import List, Tuple

from vllm import EngineArgs, AsyncLLMEngine, RequestOutput, SamplingParams, TokensPrompt
from vllm.utils import FlexibleArgumentParser

import time, torch


def create_test_prompts() -> List[Tuple[str, SamplingParams]]:
    """Create a list of test prompts with their sampling parameters."""
    torch.cuda.set_sync_debug_mode("warn")
    return [
        (
            {"prompt_token_ids":[100 for _ in range(1000)]},
            SamplingParams(temperature=0.0, max_tokens=300, ignore_eos=True),
        )
        for _ in range(120)
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


async def process_requests(engine: AsyncLLMEngine, test_prompts: List[Tuple[str, SamplingParams]]):
    """Continuously process a list of prompts and handle the outputs."""
    global request_id
    request_id = 0
    # first we add all the prompts to the engine

    prompt, sampling_params = test_prompts.pop(0)
    last_time = time.time()
    async def put_one_and_fetch(prompt, sampling_params):
        global request_id
        request_id += 1
        async for output in engine.generate(prompt, sampling_params, str(request_id)):
            pass
        print("once")

    asyncio.create_task(put_one_and_fetch(prompt, sampling_params))
    last_time = time.time()

    start_time = time.time()
    hit_count = 0
    while True:
        
        # request_outputs: List[RequestOutput] = engine.step()
        to_sleep = 1
        await asyncio.sleep(to_sleep)
        print("put once")
        asyncio.create_task(put_one_and_fetch(prompt, sampling_params))
        # for request_output in request_outputs:
        #     if request_output.finished:
        #         print(request_output.request_id, request_output.outputs[0].text[:50])


def initialize_engine(args: argparse.Namespace) -> AsyncLLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return AsyncLLMEngine.from_engine_args(engine_args)

@torch.inference_mode()
async def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine(args)
    test_prompts = create_test_prompts()
    await process_requests(engine, test_prompts)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Demo on using the LLMEngine class directly"
    )
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    import asyncio
    asyncio.run(main(args))
