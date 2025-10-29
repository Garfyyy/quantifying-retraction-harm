import os
import glob
import matplotlib

matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

# File paths list (adjusted for c1 to c6)
data_c1_csv_list = [
    './2jif/paper_c1/0-3/',
    './2jif/paper_c1/3-5/',
    './2jif/paper_c1/5-10/',
    './2jif/paper_c1/10-20/',
    './2jif/paper_c1/20-inf/',
]
# Repeat for c2 to c6, same as c1 in this case
data_c2_csv_list = [
    './2jif/paper_c2/jif-D/0-3/',
    './2jif/paper_c2/jif-D/3-5/',
    './2jif/paper_c2/jif-D/5-10/',
    './2jif/paper_c2/jif-D/10-20/',
    './2jif/paper_c2/jif-D/20-inf/',
]
data_c3_csv_list = [
    './2jif/paper_c3/jif-D/0-3/',
    './2jif/paper_c3/jif-D/3-5/',
    './2jif/paper_c3/jif-D/5-10/',
    './2jif/paper_c3/jif-D/10-20/',
    './2jif/paper_c3/jif-D/20-inf/',
]
data_c4_csv_list =  [
    './2jif/paper_c4/jif-D/0-3/',
    './2jif/paper_c4/jif-D/3-5/',
    './2jif/paper_c4/jif-D/5-10/',
    './2jif/paper_c4/jif-D/10-20/',
    './2jif/paper_c4/jif-D/20-inf/',
]
data_c5_csv_list = [
    './2jif/paper_c5/jif-D/0-3/',
    './2jif/paper_c5/jif-D/3-5/',
    './2jif/paper_c5/jif-D/5-10/',
    './2jif/paper_c5/jif-D/10-20/',
    './2jif/paper_c5/jif-D/20-inf/',
]
data_c6_csv_list = [
    './2jif/paper_c6/jif-D/0-3/',
    './2jif/paper_c6/jif-D/3-5/',
    './2jif/paper_c6/jif-D/5-10/',
    './2jif/paper_c6/jif-D/10-20/',
    './2jif/paper_c6/jif-D/20-inf/',
]

# colors = ['#A52A2A', '#9370DB', '#5F9EA0', '#008000', '#8B0000']
colors = ['#BC8F8F', '#8B4513', '#556B2F','#D2691E','#2F4F4F']

markers = ['o', 's', 'D', '^', 'P']  # 自定义标记样式：圆形、方形、菱形、三角形、五边形
# linestyles = ['-', '--', '-.', ':', '-.']  # 使用不同的线型样式（实线、虚线、点划线等）
linestyles = ['-', '--', '-.', (0, (10, 5)), (0, (1, 1))]

# Function to read second row (index 1)
def read_second_row(directory_path):
    csv_files = glob.glob(os.path.join(directory_path, '*.csv'))

    if len(csv_files) != 1:
        raise ValueError(f"Expected exactly one CSV file in {directory_path}, but found {len(csv_files)}.")
    data = np.loadtxt(csv_files[0], delimiter=",")

    # Return the second row (median) scaled by 100
    return data[1] * 100

# Create a canvas and gridspec for 6 individual plots
fig, axes = plt.subplots(3, 2, figsize=(28, 30))


# Update font style for plots
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.weight': 'semibold',
    'axes.labelweight': 'semibold',
    'axes.titleweight': 'semibold',
})




# List of CSV lists for c1 to c6
data_csv_lists = [data_c1_csv_list, data_c2_csv_list, data_c3_csv_list,
                  data_c4_csv_list, data_c5_csv_list, data_c6_csv_list]



for idx, data_csv_list in enumerate(data_csv_lists):
    ax = axes[idx // 2, idx % 2]  # Determine the correct subplot location
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    max_value = 0  # Initialize the maximum value for y-axis scaling
    min_value = float('inf')  # Initialize the minimum value for y-axis scaling

    # Prepare data for the plot (5 lines)
    for i, data_csv in enumerate(data_csv_list):
        q2 = read_second_row(data_csv)  # Get the second row (median) for the CSV file
        max_value = max(max_value, q2.max())  # Update the maximum value
        min_value = min(min_value, q2.min())  # Update the minimum value
        positions = np.arange(1, len(q2) + 1)

        # Plot the line with updated labels
        range_labels = ["IF: 0-3", "IF: 3-5", "IF: 5-10", "IF: 10-20", "IF: 20-inf"]
        ax.plot(
            positions, q2, label=range_labels[i],
            linewidth=3, color=colors[i],
            linestyle=linestyles[i],  # 设置每条线的不同线型
            markersize=10, markerfacecolor='white', markeredgewidth=1
        )

        ax.scatter(
            positions, q2,
            color=colors[i], marker=markers[i], s=100, edgecolors='white', linewidths=1
        )

    # Set dynamic y-axis limit with a 10% margin
    y_min = min_value * 0.9
    y_max = max_value * 1.1
    ax.set_ylim(0, 65)

    # Title, axis labels, and formatting
    # ax.set_title(f'Paper_c{idx + 1} All Fields', fontsize=35)
    ax.set_xlabel('Years after publication', fontsize=35)
    ax.set_ylabel('Harm value (%)', fontsize=35)

    # Axis ticks formatting
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
    tick_spacing = (65 - 0) / 5
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))  # Dynamic tick spacing
    ax.tick_params(axis='both', which='major', labelsize=35)
    ax.tick_params(axis='both', which='minor', labelsize=35)

    # Add legend
    ax.legend(loc='upper left', ncol=1, columnspacing=1, handletextpad=0.5, fontsize=23)

    # Add label (a-f) in the top-left corner of each subplot
    # labels = ['a', 'b', 'c', 'd', 'e', 'f']
    labels = ['A', 'B', 'C', 'D', 'E','F']

    ax.text(0, 1.1, labels[idx], transform=ax.transAxes, fontsize=50, fontweight='bold', va='top', ha='left')


# Adjust layout to prevent overlapping
plt.tight_layout()

# Save the plot with tight layout
plt.savefig('fig4_final.pdf', format='pdf', bbox_inches='tight')
