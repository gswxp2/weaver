import os
import time
dataset_name = os.sys.argv[1]
def check_weaver(timeout=60):
    start = time.time() 
    while True:
        ret = os.system("curl http://localhost:8000/v1/completions 2>/dev/null 1>/dev/null")
        if ret == 0:
            return True 
        time.sleep(1)
        if time.time() - start > timeout:
            # os.system("killall -9 python3")
            return False
def check_and_start_weaver(dedicate, output_prefix):
    # os.system("killall -9 python3")
    # os.system("killall -9 python3")
    # os.system("killall -9 python3")
    if check_weaver(5):
        return
    if dedicate:
        os.system(f"bash start_weaver_single.sh {output_prefix}")
    else:
        os.system(f"bash start_weaver.sh {output_prefix}")
    start = time.time()
    while True:
        res = check_weaver(60)
        if res:
            break
        assert res, "weaver not started"

# rates = [10]
# for i in range(1,14,2):
#     rates.append(i)
# for i in range(14,20,1):
#     rates.append(i)
if dataset_name == "azure":
    qpss = [ i for i in range(1, 15, 3)]
    qpss.extend([i for i in range(15, 23, 1)])
if dataset_name == "burst":
    qpss = [ i for i in range(1, 11, 3)]
    qpss.extend([i for i in range(12, 18, 1)])
for dedicate in [False, True]:
    for qps in qpss:
        prefix = "res_weaver"
        output_prefix = f"{dataset_name}/res_weaver/{qps}_{dedicate}"
        os.system(f"mkdir -p {dataset_name}/res_weaver")
        check_and_start_weaver(dedicate, output_prefix)
        command = f"/home/weaverae/miniforge3/envs/weaver/bin/python3 ~/weaver/ae_exps/scripts/benchmarks/benchmark_serving.py     --backend vllm     --dataset-name azure \
        --model meta-llama/Meta-Llama-3-8B   --dataset-path  ~/weaver/{dataset_name}.json --num-prompts {qps*200}     \
        --endpoint /v1/completions    --tokenizer meta-llama/Meta-Llama-3-8B \
        --save-result   --request-rate {qps}  --ignore-eos > {dataset_name}/res_weaver/{qps}_{dedicate}.txt"
        print(command)
        os.system(command)
        os.system("killall -9 python3")
        os.system("killall -9 python3")
        os.system("killall -9 python3")
    time.sleep(10)
