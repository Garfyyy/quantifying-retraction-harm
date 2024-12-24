import os
import glob
import matplotlib

matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
from matplotlib.gridspec import GridSpec

# File paths list
data_csv_list = [
    './1normal/normal-ND/paper_c2/All-1374471.csv',
    './1normal/normal-ND/paper_c3/All-3529585.csv',
    './1normal/normal-ND/paper_c4/All-3342674.csv',
    './1normal/normal-ND/paper_c5/All-2397650.csv',
    './1normal/normal-ND/paper_c6/All-1762079.csv'
]

# Create a canvas and gridspec (3 rows, 2 columns for first two rows, third row spans all columns but centered)
fig = plt.figure(figsize=(80, 80))
gs = GridSpec(3, 2, figure=fig)

color1 = '#556B2F'  # Green for Harm < 0.0
color2 = '#FFA500'  # Orange for Harm >= 0

# Update font style for plots
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.weight': 'normal',
    'axes.labelweight': 'normal',
    'axes.titleweight': 'semibold',
})

# Function to determine color based on value
def get_color(x: float) -> str:
    if x < 0.0:
        return color1
    elif x >= 0:
        return color2

# Labels for subplots
# subplot_labels = ['a', 'b', 'c', 'd', 'e']
subplot_labels = ['A', 'B', 'C', 'D', 'E']

# Plot data for each CSV file
for idx, data_csv in enumerate(data_csv_list):
    data = np.loadtxt(data_csv, delimiter=",")
    q3 = data[0] * 100  # Upper quartile
    q2 = data[1] * 100  # Median
    q1 = data[2] * 100  # Lower quartile
    # positions = np.arange(1, len(q2) + 1) + 1
    positions = np.arange(1, 11, 1)  # 固定为 2 到 11 的范围

    # Determine the row and column for the subplot
    if idx == 4:  # Special case for c6, span third row with custom placement
        ax = fig.add_axes([0.28, 0.04, 0.5, 0.28])  # Manually set position to center

    else:
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs[row, col])  # Regular placement
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Plotting each box
    for pos, lower, upper, mid_v in zip(positions, q1, q3, q2):
        segments = [
            ((-np.inf, 0), color1),
            ((0, np.inf), color2)
        ]

        for (segment_min, segment_max), color in segments:
            y_bottom = np.clip(lower, segment_min, segment_max)
            y_top = np.clip(upper, segment_min, segment_max)

            mask = y_top > y_bottom
            if np.any(mask):
                ax.vlines(pos, y_bottom, y_top, colors=color, linewidth=18)

        ax.hlines(lower, pos - 0.15, pos + 0.15, colors=get_color(lower), linewidth=18)
        ax.hlines(upper, pos - 0.15, pos + 0.15, colors=get_color(upper), linewidth=18)

    # Plot median line
    median_line, = ax.plot(positions, q2, marker='^', color='#8B4513', markersize=40, linestyle='--', label='Median', linewidth=15)

    # Add legend for the current subplot
    legend_elements = [
        Line2D([0], [0], color=color1, label='Harm < 0.0', linewidth=9),
        Line2D([0], [0], color=color2, label='0.0 ≤ Harm', linewidth=9),
        median_line
    ]
    ax.legend(
        handles=legend_elements,
        loc='upper left',
        ncol=1,
        columnspacing=1,
        handletextpad=0.5,
        fontsize=90
    )

    # Add subplot label in the top-left corner
    ax.text(0, 1.1, subplot_labels[idx], transform=ax.transAxes, fontsize=100, fontweight='bold', va='top', ha='left',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    # Title, axis labels and adjust font size
    # ax.set_title(f'Paper_c{idx + 2} All Fields', fontsize=37, pad=20)
    ax.set_xlabel('Years after publication', fontsize=90)
    ax.set_ylabel('Harm value (%)', fontsize=90)

    # Axis range and ticks
    # ax.set_ylim(top=300)
    ax.set_ylim(-50, 300)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.tick_params(axis='both', which='major', labelsize=90)
    ax.tick_params(axis='both', which='minor', labelsize=90)

    # Inset axis
    inset_ax = inset_axes(ax, width="40%", height="40%", loc='upper right',
                          bbox_to_anchor=(-0.18, -0.15, 1.13, 1.13), bbox_transform=ax.transAxes)
    inset_ax.plot(positions, q2, marker='^', color='#8B4513', markersize=60, linestyle='--', linewidth=15)
    inset_ax.spines['top'].set_visible(False)
    inset_ax.spines['right'].set_visible(False)
    # Adjust inset y-limits dynamically
    padding = 6
    inset_ax.set_xlim(1, len(q2) )
    inset_ax.set_ylim(0, 110)  # Set the x-axis limits to fixed range of 20 to 70
    tick_spacing = (110 - 0) / 5
    inset_ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))  # Dynamic tick spacing
    inset_ax.xaxis.set_major_locator(ticker.MultipleLocator(1))  # 间隔为 1

    # inset_ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    inset_ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
    inset_ax.tick_params(axis='both', which='major', labelsize=90)

    # Connection lines between main plot and inset
    for i, pos in enumerate(positions):
        if i == 0 or i == len(positions) - 1:
            con = ConnectionPatch(
                xyA=(pos, q2[i]), xyB=(pos, q2[i]), coordsA="data", coordsB="data",
                axesA=ax, axesB=inset_ax, color="black", lw=6, linestyle=(0, (10, 5))  # 10 像素线段，5 像素间距
            )

            fig.add_artist(con)

# Adjust layout to prevent overlapping
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# Save the plot with tight layout
plt.savefig('fig_s1_nd_final.pdf', format='pdf', bbox_inches='tight', pad_inches=0.2)
