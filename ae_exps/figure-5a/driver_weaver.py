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
def check_and_start_weaver(solo, output_prefix, client_qps):
    os.system("killall -9 python3")
    os.system("killall -9 python3")
    os.system("killall -9 python3")
    if check_weaver(5):
        return
    if solo:
        os.system(f"bash start_weaver_single.sh {output_prefix}")
    else:
        os.system(f"bash start_weaver.sh {output_prefix} {client_qps}")
    start = time.time()
    while True:
        res = check_weaver(120)
        if res:
            break
        return 
        # assert res, "weaver not started"
            
            
for solo in [False]:
    for client_qps in [1,2,3.3,5]:
        prefix = "res_weaver"
        os.system(f"mkdir -p {prefix}")
        # for methods in ["blocking","cpu-control-async","multi-stream","GPU-control","GPU-control-chunk"]
        output_prefix = f"{prefix}/{client_qps}_{solo}"
        check_and_start_weaver(solo, output_prefix, client_qps)
        command = f"python3 ~/weaver/vllm/benchmarks/benchmark_serving.py     --backend vllm     --dataset-name azure \
        --model meta-llama/Meta-Llama-3-8B   --dataset-path  ~/weaver/burst.json --num-prompts {10*200}     \
        --endpoint /v1/completions     --tokenizer meta-llama/Meta-Llama-3-8B --random-output-len 256 \
        --save-result   --request-rate {10}  --ignore-eos > {prefix}/{client_qps}_{solo}.txt"
        print(command)
        os.system(command)
        os.system("killall -9 python3")
        os.system("killall -9 python3")
        os.system("killall -9 python3")
    time.sleep(10)
