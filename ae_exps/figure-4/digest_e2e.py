import os
import re

dataset_name = os.sys.argv[1]
def digest_mux(filename):
    with open(filename, "r") as f:
        text = f.read()
        # 正则表达式提取数据
        pattern = re.compile(
        r"Model: (?P<model>\S+)\n"
        r".*?(?P<sched>nosched|sched) Average latency: (?P<latency>\d+\.\d+) s\n"
        r".*?Average latency per token: (?P<latency_per_token>\d+\.\d+) s\n"
        r".*?Average latency per output token: (?P<latency_per_output_token>\d+\.\d+) s",
        re.DOTALL
        )

        # 提取匹配结果
        data = []
        for match in pattern.finditer(text):
            data.append({
                "model": match.group("model"),
                "sched": match.group("sched"),
                "latency": float(match.group("latency")),
                "latency_per_token": float(match.group("latency_per_token")),
                "latency_per_output_token": float(match.group("latency_per_output_token")),
            })
    # we compare the latency of llm-0 sched/nosched for overload
    overload = False
    for d in data:
        if d["model"] == "llm-0":
            if d["sched"] == "nosched":
                nosched = d["latency"]
            else:
                sched = d["latency"]
    # print(nosched, sched)
    if nosched>sched+1:
        overload = True
        return overload, 1000, 1000
    for d in data:
        if d["model"] == "llm-1":
            if d["sched"] == "nosched":
                nosched = d["latency"]
            else:
                sched = d["latency"]
    if nosched>sched+1:
        overload = True
        return overload, 1000, 1000
    # print(nosched, sched)
    # return the latency of sched llm-0 and llm-1, in order
    for d in data:
        if d["model"] == "llm-0" and d["sched"] == "nosched":
            sched_llm_0 = d["latency_per_output_token"]
        if d["model"] == "llm-1" and d["sched"] == "nosched":
            sched_llm_1 = d["latency_per_output_token"]
    return overload, sched_llm_0*1000, sched_llm_1*1000

def parse_benchmark(text):
    data = {}
    current_section = None

    # 定义正则表达式来匹配指标行，例如 "指标名称:    数值"
    pattern = re.compile(r'^(?P<key>[^\:]+):\s+(?P<value>[0-9.]+)')

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('===') or line.startswith('---'):
            continue  # 跳过分隔线和空行

        match = pattern.match(line)
        if match:
            key = match.group('key').strip()
            value = match.group('value').strip()

            # 如果需要，可以进一步处理key和value，例如移除单位
            # 这里我们保留原始的key和数值字符串
            data[key] = value

    return data

def digest_weaver(preix):
    filename = f"{preix}.txt"
    overload = False
    with open(filename, "r") as f:
        text = f.read()
        try:
            output = parse_benchmark(text)
            overload = float(output['Median TTFT (ms)']) > 500 or output['Successful requests'] == '0'
            sender = float(output['Mean TPOT (ms)'])
        except:
            print(filename)
        if overload:
            return overload, 1000, 1000
    filename = f"{preix}-receiver.log"
    if not os.path.exists(filename):
        return overload, sender, None
    with open(filename, "r") as f:
        text = f.read()
        # get all lines with ModelRunner: 17.46xxx
        lines = [line for line in text.split("\n") if "ModelRunner" in line]
        # get the middle 20% of the lines
        lines = lines[len(lines)*3//5:len(lines)*4//5]
        # get the average of the middle 20%
        lines = [float(line.split(":")[1].strip()) for line in lines]
        return overload, sender, sum(lines)/len(lines)
        # return float(output['Mean TPOT (ms)']), overload
data_mux = []
datas = []
for i in range(1,25,1):
    datas.append(i)
datas.sort()
for qps in datas:
    try:
        data_mux.append([qps,digest_mux(f"{dataset_name}/res_mux_mps/{qps}.txt")])
    except:
        continue
    # print(qps,digest_mux(f"res_mux_mps_azure/{qps}.txt"))
data_mux_temporal = []
for qps in datas:
    try:
        data_mux_temporal.append([qps,digest_mux(f"{dataset_name}/res_mux_temporal/{qps}.txt")])   
    except:
        continue
data_weaver = []
data_weaver_dedicate = []
for qps in datas:
    try:
        data_weaver_dedicate.append([qps, digest_weaver(f"{dataset_name}/res_weaver/{qps}_True")])
    except:
        continue
for qps in datas:
    try:
        data_weaver.append([qps, digest_weaver(f"{dataset_name}/res_weaver/{qps}_False")])
    except:
        continue
# print(data_weaver)
import matplotlib.pyplot as plt
plt.figure()
plt.plot([x[0] for x in data_mux], [x[1][1] for x in data_mux], label="muxserve_mps")
plt.plot([x[0] for x in data_mux_temporal], [x[1][1] for x in data_mux_temporal], label="muxserve_temporal")
plt.plot([x[0] for x in data_weaver], [x[1][1] for x in data_weaver], label="weaver")
plt.plot([x[0] for x in data_weaver_dedicate], [x[1][1] for x in data_weaver_dedicate], label="weaver_dedicate")
plt.legend()
# limit y by 60
plt.ylim(0,60)
plt.xlabel("QPS")
plt.ylabel("TPOT (ms)")
plt.title("cold start")
plt.savefig(f"{dataset_name}_hot.png")
plt.figure()
plt.plot([x[0] for x in data_mux], [x[1][2] for x in data_mux], label="muxserve_mps")
plt.plot([x[0] for x in data_mux_temporal], [x[1][2] for x in data_mux_temporal], label="muxserve_temporal")
plt.plot([x[0] for x in data_weaver], [x[1][2] for x in data_weaver], label="weaver")
plt.plot([x[0] for x in data_weaver_dedicate], [data_weaver_dedicate[0][1][1] for x in data_weaver_dedicate], label="weaver_dedicate")
plt.ylim(0,60)
plt.xlabel("QPS")
plt.ylabel("TPOT (ms)")
plt.legend()
plt.savefig(f"{dataset_name}_cold.png")
for item in data_mux:
    print(item[0], item[1][0], item[1][1], item[1][2])
print("=======================================")
for item in data_mux_temporal:
    print(item[0], item[1][0], item[1][1], item[1][2])
print("=======================================")
for item in data_weaver_dedicate:
    print(item[0], item[1][0], item[1][1], item[1][2])
print("=======================================")
for item in data_weaver:
    print(item[0], item[1][0], item[1][1], item[1][2])
    
