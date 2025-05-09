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
datas = [1,2,3.3,5]
datas.sort()
for qps in datas:
    try:
        data_mux.append([qps,digest_mux(f"res_mux_mps_burst/{qps}.txt")])
    except:
        continue
    print(qps,digest_mux(f"res_mux_mps_burst/{qps}.txt"))
data_mux_temporal = []
for qps in datas:
    try:
        data_mux_temporal.append([qps,digest_mux(f"res_mux_temporal_burst/{qps}.txt")])   
    except:
        continue
data_weaver = []
for qps in datas:
    try:
        data_weaver.append([qps, digest_weaver(f"res_weaver/{qps}_False")])
    except:
        continue
print(data_weaver)
print(data_mux_temporal)
print(data_mux)
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np

# Placeholder data (approximated from the image)
# Categories for the x-axis
categories = ['10:1', '5:1', '3:1', '2:1']
n_categories = len(categories)

# Data for "Hot" plot
muxserve_hot_data = [int(x[1][1]) for x in data_mux]
muxtemporal_hot_data = [int(x[1][1]) for x in data_mux_temporal]
weaver_hot_data = [int(x[1][1]) for x in data_weaver]

# Data for "Cold" plot
muxserve_cold_data = [int(x[1][2]) for x in data_mux]
muxtemporal_cold_data = [int(x[1][2]) for x in data_mux_temporal]
weaver_cold_data = [int(x[1][2]) for x in data_weaver]
# --- Plotting ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5), sharey=True)

bar_width = 0.25  # Width of a single bar
index = np.arange(n_categories) # x locations for the groups

# Colors (approximating the image)
color_muxserve = '#E6F0FF' # Very light blue, almost white
color_muxtemporal = '#A8D8F8' # Light blue
color_weaver = '#367FB2'    # Darker blue

# --- Plot 1: Hot ---
rects1_hot = ax1.bar(index - bar_width, muxserve_hot_data, bar_width,
                     label='MuxServe', color=color_muxserve, edgecolor='black')
rects2_hot = ax1.bar(index, muxtemporal_hot_data, bar_width,
                     label='Mux-Temporal', color=color_muxtemporal, edgecolor='black')
rects3_hot = ax1.bar(index + bar_width, weaver_hot_data, bar_width,
                     label='Weaver', color=color_weaver, edgecolor='black')

ax1.set_title('Hot', fontsize=14, loc='left', y=0.92, x=0.05) # Adjusted title position
ax1.set_ylabel('TPOT (ms)', fontsize=14)
ax1.set_xticks(index)
ax1.set_xticklabels(categories, fontsize=12)
ax1.tick_params(axis='y', labelsize=12)
ax1.set_ylim(0, 48)

# Add horizontal dotted lines for reference
for y_val in [20, 40]:
    ax1.axhline(y=y_val, color='grey', linestyle=':', linewidth=0.8, dashes=(1, 3)) # Finer dots

# --- Plot 2: Cold ---
rects1_cold = ax2.bar(index - bar_width, muxserve_cold_data, bar_width,
                      color=color_muxserve, edgecolor='black')
rects2_cold = ax2.bar(index, muxtemporal_cold_data, bar_width,
                      color=color_muxtemporal, edgecolor='black')
rects3_cold = ax2.bar(index + bar_width, weaver_cold_data, bar_width,
                      color=color_weaver, edgecolor='black')

ax2.set_title('Cold', fontsize=14, loc='left', y=0.92, x=0.05) # Adjusted title position
ax2.set_xticks(index)
ax2.set_xticklabels(categories, fontsize=12)
# ax2.tick_params(axis='y', labelsize=12) # Not needed due to sharey=True
ax2.set_ylim(0, 48) # Ensure same y-axis limits

for y_val in [20, 40]:
    ax2.axhline(y=y_val, color='grey', linestyle=':', linewidth=0.8, dashes=(1, 3))

# --- Legend and Labels ---
# Create legend handles and labels (can take from either plot, e.g., 'Hot' plot)
handles = [rects1_hot, rects2_hot, rects3_hot]
labels = ['MuxServe', 'Mux-Temporal', 'Weaver']
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=3, fontsize=12, frameon=False)

# Overall X-axis label for the figure
fig.supxlabel('Hot-to-Cold Request Ratio', fontsize=14, y=0.03)

# Subplot label (a)
fig.text(0.5, -0.04, '(a)', ha='center', va='bottom', fontsize=18, fontweight='bold', transform=fig.transFigure)

# Adjust layout to prevent overlap and make space for legend/labels
plt.tight_layout(rect=[0, 0.05, 1, 0.92]) # rect=[left, bottom, right, top]

plt.show()

plt.savefig("figure_5a.png", dpi=300, bbox_inches='tight')