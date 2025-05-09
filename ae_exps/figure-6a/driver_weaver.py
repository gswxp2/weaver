import os
import time
def check_weaver(timeout=1200):
    start = time.time()
    while True:
        ret = os.system("curl http://localhost:8000/v1/completions 2>/dev/null 1>/dev/null")
        if ret == 0:
            return True 
        time.sleep(1)
        if time.time() - start > timeout:
            os.system("killall -9 python3")
            return False
def check_and_start_weaver(solo, output_prefix):
    os.system("killall -9 python3")
    os.system("killall -9 python3")
    os.system("killall -9 python3")
    if check_weaver(5):
        return
    if solo:
        os.system(f"bash start_weaver_single.sh {output_prefix}")
    else:
        os.system(f"bash start_weaver.sh {output_prefix}")
    start = time.time()
    while True:
        res = check_weaver(120)
        if res:
            break
        assert res, "weaver not started"

solo = False 
for qps in [15]:
    for strategy in ["none"]:
        prefix = "res_weaver"
        os.system(f"mkdir -p {prefix}")
        # for methods in ["blocking","cpu-control-async","multi-stream","GPU-control","GPU-control-chunk"]
        output_prefix = f"{prefix}/{strategy}_{qps}_{solo}"
        # remove ABAL_USE_V0 and chunk_enabled from the environ
        os.environ.pop("ABAL_USE_V0", None)
        os.environ.pop("CHUNKE_ENABLED", None)
        if strategy == "none":
            os.environ["ABAL_USE_V0"] = "1"
            os.environ["CHUNKE_ENABLED"] = "0"
        elif strategy == "cpu":
            os.environ["ABAL_USE_V0"] = "0"
            os.environ["CHUNKE_ENABLED"] = "0"
        elif strategy == "weaver":
            os.environ["ABAL_USE_V0"] = "0"
            os.environ["CHUNKE_ENABLED"] = "1"
        check_and_start_weaver(solo, output_prefix)
        command = f"python3 ~/weaver/vllm/benchmarks/benchmark_serving.py     --backend vllm     --dataset-name azure \
        --model meta-llama/Meta-Llama-3-8B   --dataset-path  ~/weaver/azure.json --num-prompts {qps*200}     \
        --endpoint /v1/completions     --tokenizer meta-llama/Meta-Llama-3-8B --random-output-len 256 \
        --save-result   --request-rate {qps}  --ignore-eos > {prefix}/{strategy}_{qps}_{solo}.txt"
        print(command)
        os.system(command)
        os.system("killall -9 python3")
        os.system("killall -9 python3")
        os.system("killall -9 python3")
    time.sleep(10)
