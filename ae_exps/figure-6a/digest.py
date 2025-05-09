import os
import re
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
            overload = float(output['Median TTFT (ms)']) > 1000 or output['Successful requests'] == '0'
            sender = float(output['Mean TPOT (ms)'])
        except Exception as e:
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
datas = [5, 10,15]
datas.sort()
data_none = []
data_cpu = []
data_weaver = []
for qps in datas:
    try:
        data_none.append([qps, digest_weaver(f"res_weaver/none_{qps}_False")])
    except:
        continue
for qps in datas:
    try:
        data_cpu.append([qps, digest_weaver(f"res_weaver/cpu_{qps}_False")])
    except Exception as e:
        continue
for qps in datas:
    try:
        data_weaver.append([qps, digest_weaver(f"res_weaver/weaver_{qps}_False")])
    except:
        continue
print(data_none)
print(data_cpu)
print(data_weaver)
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np

# --- Data (Estimated from the image) ---
labels_qps = ['5', '10', '15']
x_indices = np.arange(len(labels_qps)) # 0, 1, 2

# Hot Model Data
cpu_control_hot_actual = [ x[1][1] for x in data_none]
gpu_control_hot = [x[1][1] for x in data_cpu]
op_splitting_hot = [x[1][1] for x in data_weaver]

# Cold Model Data
cpu_control_cold = [x[1][2] for x in data_none]
gpu_control_cold = [x[1][2] for x in data_cpu]
op_splitting_cold = [x[1][2] for x in data_weaver]

# --- Plotting Parameters ---
bar_width = 0.25
# Colors (approximating the image: light lavender, light blue, darker blue)
color_cpu = '#E6E6FA' # Lavender for CPU Control
color_gpu = '#ADD8E6' # Light blue for +GPU Control
color_op = '#4682B4'  # Steel blue for +Op Splitting

legend_labels = ['CPU Control', '+GPU Control', '+Op Splitting']

# --- Create Figure and Axes ---
# We'll use subplots to create the two distinct "Hot" and "Cold" sections
fig, axes = plt.subplots(1, 2, figsize=(10, 5.5), sharey=False) # sharey=False as y-scales differ

# --- Plot 1: Hot Model ---
ax_hot = axes[0]
y_limit_hot = 70  # Visual y-limit for the 'Hot' plot, actual data for first bar is higher

# Clip the first CPU control bar for display, annotation will show actual value
cpu_control_hot_display = np.copy(cpu_control_hot_actual).astype(float)
if cpu_control_hot_display[0] > y_limit_hot:
    cpu_control_hot_display[0] = y_limit_hot

bar1_hot = ax_hot.bar(x_indices - bar_width, cpu_control_hot_display, bar_width,
                      label=legend_labels[0], color=color_cpu, edgecolor='grey')
bar2_hot = ax_hot.bar(x_indices, gpu_control_hot, bar_width,
                      label=legend_labels[1], color=color_gpu, edgecolor='grey')
bar3_hot = ax_hot.bar(x_indices + bar_width, op_splitting_hot, bar_width,
                      label=legend_labels[2], color=color_op, edgecolor='grey')



ax_hot.set_ylabel('TPOT (ms)', fontsize=14)
ax_hot.set_xlabel('Hot Model Request Rate (qps)', fontsize=14)
ax_hot.set_title('Hot', fontsize=16, y=1.02) # y to give a bit space from legend
ax_hot.set_xticks(x_indices)
ax_hot.set_xticklabels(labels_qps, fontsize=12)
ax_hot.set_yticks(np.arange(0, 80, 20)) # 0, 20, 40, 60
ax_hot.set_ylim(0, y_limit_hot + 15) # Extend ylim a bit for annotation space
ax_hot.tick_params(axis='y', labelsize=12)
ax_hot.grid(axis='y', linestyle=':', linewidth=0.7, alpha=0.7)

# --- Plot 2: Cold Model ---
ax_cold = axes[1]
y_limit_cold = 40

bar1_cold = ax_cold.bar(x_indices - bar_width, cpu_control_cold, bar_width,
                        color=color_cpu, edgecolor='grey')
bar2_cold = ax_cold.bar(x_indices, gpu_control_cold, bar_width,
                        color=color_gpu, edgecolor='grey')
bar3_cold = ax_cold.bar(x_indices + bar_width, op_splitting_cold, bar_width,
                        color=color_op, edgecolor='grey')

#

ax_cold.set_xlabel('Hot Model Request Rate (qps)', fontsize=14) # As per original image, label is the same
ax_cold.set_title('Cold', fontsize=16, y=1.02)
ax_cold.set_xticks(x_indices)
ax_cold.set_xticklabels(labels_qps, fontsize=12)
ax_cold.set_yticks(np.arange(0, 41, 20)) # 0, 20, 40
ax_cold.set_ylim(0, y_limit_cold + 5)
ax_cold.tick_params(axis='y', labelsize=12)
ax_cold.grid(axis='y', linestyle=':', linewidth=0.7, alpha=0.7)


# --- Legend (Common for both subplots, placed above) ---
# Get handles and labels from one of the plots (e.g., ax_hot)
handles = [bar1_hot, bar2_hot, bar3_hot]
fig.legend(handles, legend_labels, loc='upper center',
           bbox_to_anchor=(0.5, 0.98), # Position legend above plots
           ncol=3, frameon=False, fontsize=12)

# --- Final Adjustments ---
plt.tight_layout(rect=[0, 0, 1, 0.92]) # rect=[left, bottom, right, top] to make space for legend

# Show plot
plt.show()
plt.savefig("figure-6a.png", dpi=300, bbox_inches='tight')