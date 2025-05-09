import matplotlib.pyplot as plt
import numpy as np

# Placeholder data (approximated from the image)
# Categories for the x-axis
categories = ['10:1', '5:1', '3:1', '2:1']
n_categories = len(categories)

# Data for "Hot" plot
muxserve_hot_data = np.array([31, 32, 31, 32])
muxtemporal_hot_data = np.array([35, 35.5, 39, 44])
weaver_hot_data = np.array([27, 27, 27, 28])

# Data for "Cold" plot
muxserve_cold_data = np.array([16, 17, 19, 20.5])
muxtemporal_cold_data = np.array([33, 35, 39, 44])
weaver_cold_data = np.array([20, 21, 24, 25])

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