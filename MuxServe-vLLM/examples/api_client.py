"""Example Python client for vllm.entrypoints.api_server"""

import argparse
import asyncio
import json
from typing import Iterable, List, Dict, Any

import requests
import aiohttp


def clear_line(n: int = 1) -> None:
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)


async def async_post_http_request(
        prompt: str,
        api_url: str,
        max_tokens: int = 2048,
        request_id: str = None) -> Dict[str, Any]:
    headers = {"User-Agent": "Test Client"}

    pload = {
        "prompt": prompt,
        "n": 1,
        "use_beam_search": False,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if request_id is not None:
        pload["request_id"] = request_id
    async with aiohttp.ClientSession() as session:
        async with session.post(api_url,
                                headers=headers,
                                json=pload) as response:
            data = await response.json()
    return data


async def get_async_response(prompt: str, api_url: str, max_tokens: int = 1, request_id: str = None):
    response = await async_post_http_request(prompt, api_url, max_tokens, request_id)
    return response["text"]


def post_http_request(prompt: str,
                      api_url: str,
                      n: int = 1,
                      stream: bool = False,
                      max_tokens: int = 2048) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    pload = {
        "prompt": prompt,
        "n": n,
        "use_beam_search": n > 1,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    response = requests.post(api_url, headers=headers, json=pload, stream=True)
    return response


def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["text"]
            yield output


def get_response(response: requests.Response) -> List[str]:
    data = json.loads(response.content)
    output = data["text"]
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--prompt", type=str, default="San Francisco is a")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--request-id", type=str, default=None)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--async-server", action="store_true")
    args = parser.parse_args()
    prompt = args.prompt
    api_url = f"http://{args.host}:{args.port}/generate"
    n = args.n
    stream = args.stream
    async_server = args.async_server
    max_tokens = args.max_tokens
    request_id = args.request_id

    print(f"Request id: {request_id}")
    print(f"Prompt: {prompt!r}\n", flush=True)

    if async_server:
        assert stream is False
        response = asyncio.run(get_async_response(prompt, api_url, max_tokens, request_id))
    else:
        response = post_http_request(prompt, api_url, n, stream, max_tokens)

    if stream:
        num_printed_lines = 0
        for h in get_streaming_response(response):
            clear_line(num_printed_lines)
            num_printed_lines = 0
            for i, line in enumerate(h):
                num_printed_lines += 1
                print(f"Beam candidate {i}: {line!r}", flush=True)
    else:
        output = response if async_server else get_response(response)
        for i, line in enumerate(output):
            print(f"Beam candidate {i}: {line!r}", flush=True)
