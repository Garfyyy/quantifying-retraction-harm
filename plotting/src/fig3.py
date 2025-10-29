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
    './1normal/normal-D/paper_c2/All-1469564.csv',
    './1normal/normal-D/paper_c3/All-4972207.csv',
    './1normal/normal-D/paper_c4/All-8319030.csv',
    './1normal/normal-D/paper_c5/All-10729186.csv',
    './1normal/normal-D/paper_c6/All-12500856.csv'
]

# Create a canvas and gridspec (3 rows, 2 columns for first two rows, third row spans all columns but centered)
fig = plt.figure(figsize=(16, 18))
gs = GridSpec(3, 2, figure=fig)

color1 = '#556B2F'  # Green for Harm < 0.0
color2 = '#FFA500'  # Orange for Harm >= 0

# Update font style for plots
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.weight': 'semibold',
    'axes.labelweight': 'semibold',
    'axes.titleweight': 'semibold',
})

# Function to determine color based on value
def get_color(x: float) -> str:
    if x < 0.0:
        return color1
    elif x >= 0:
        return color2

# Plot data for each CSV file
for idx, data_csv in enumerate(data_csv_list):
    data = np.loadtxt(data_csv, delimiter=",")
    q3 = data[0] * 100  # Upper quartile
    q2 = data[1] * 100  # Median
    q1 = data[2] * 100  # Lower quartile
    positions = np.arange(1, len(q2) + 1) + 1

    # Determine the row and column for the subplot
    if idx == 4:  # Special case for c6, span third row with custom placement
        ax = fig.add_axes([0.25, 0.1, 0.5, 0.25])  # Manually set position to center
    else:
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs[row, col])  # Regular placement

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
                ax.vlines(pos, y_bottom, y_top, colors=color, linewidth=3)

        ax.hlines(lower, pos - 0.15, pos + 0.15, colors=get_color(lower), linewidth=3)
        ax.hlines(upper, pos - 0.15, pos + 0.15, colors=get_color(upper), linewidth=3)

    # Plot median line
    median_line, = ax.plot(positions, q2, marker='^', color='#8B4513', markersize=8, linestyle='--', label='Median')

    # Add legend
    legend_elements = [
        Line2D([0], [0], color=color1, label='Harm < 0.0'),
        Line2D([0], [0], color=color2, label='0.0 ≤ Harm'),
        median_line
    ]
    ax.legend(
        handles=legend_elements,
        loc='upper left',
        ncol=1,
        columnspacing=1,
        handletextpad=0.5,
        fontsize=14
    )

    # Title, axis labels and adjust font size
    # ax.set_title(f'Paper_c{idx + 2} All Fields', fontsize=18)
    ax.set_xlabel('Years after publication', fontsize=16)
    ax.set_ylabel('Harm value (%)', fontsize=16)

    # Axis range and ticks
    ax.set_ylim(top=300)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)

    # Inset axis
    inset_ax = inset_axes(ax, width="40%", height="40%", loc='upper right',
                          bbox_to_anchor=(-0.14, -0.12, 1.13, 1.13), bbox_transform=ax.transAxes)
    inset_ax.plot(positions, q2, marker='^', color='#8B4513', markersize=10, linestyle='--', linewidth=2)

    # Adjust inset y-limits dynamically
    padding = 5
    inset_ax.set_xlim(2, len(q2) + 1)
    inset_ax.set_ylim(20, 70)  # Set the x-axis limits to fixed range of 20 to 70

    inset_ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    inset_ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
    inset_ax.tick_params(axis='both', which='major', labelsize=17)

    # Connection lines between main plot and inset
    for i, pos in enumerate(positions):
        if i == 0 or i == len(positions) - 1:
            con = ConnectionPatch(xyA=(pos, q2[i]), xyB=(pos, q2[i]), coordsA="data", coordsB="data",
                                  axesA=ax, axesB=inset_ax, color="black", lw=1)
            fig.add_artist(con)

    # Add label a, b, c, d, e in the top-left corner of each subplot
    labels = ['a', 'b', 'c', 'd', 'e']
    ax.text(0.05, 0.95, labels[idx], transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='left')

# Adjust layout to prevent overlapping
plt.tight_layout()

# Save the plot with tight layout
plt.savefig('paper_c1_all_in_one_centered_3x2_layout.pdf', format='pdf', bbox_inches='tight')
