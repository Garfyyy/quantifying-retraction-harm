import os
import glob
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# 获取第一个 CSV 文件
data_csv = './1normal/normal-D/paper_c1/All-119644.csv'
data = np.loadtxt(data_csv, delimiter=",")

# 将数据乘以 100
q3 = data[0] * 100
q2 = data[1] * 100
q1 = data[2] * 100
positions = np.arange(1, len(q2) + 1)

color1 = '#556B2F'  # blue
color2 = '#FFA500'  # red

# 更新字体样式
plt.rcParams.update({
    'font.family': 'Times New Roman',  # 需要系统支持这个字体，没有可以注释
    'font.weight': 'semibold',
    'axes.labelweight': 'semibold',
    'axes.titleweight': 'semibold',
})

# 创建画布和主图
fig, ax = plt.subplots(figsize=(12, 9))  # 调整画布大小，将高度减少

def get_color(x: float) -> str:
    if x < 0.0:
        return color1
    elif x >= 0:
        return color2

# 绘制数据
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
            ax.vlines(pos, y_bottom, y_top, colors=color, linewidth=5)

    ax.hlines(lower, pos - 0.15, pos + 0.15, colors=get_color(lower), linewidth=5)
    ax.hlines(upper, pos - 0.15, pos + 0.15, colors=get_color(upper), linewidth=5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
# 绘制中位数折线
median_line, = ax.plot(positions, q2, marker='^', color='#8B4513', markersize=13, linestyle='--', label='Median', linewidth=3)

# 添加图例
legend_elements = [
    Line2D([0], [0], color=color1, label='Harm < 0.0', linewidth=3),
    Line2D([0], [0], color=color2, label='0.0 ≤ Harm', linewidth=3),
    median_line
]
ax.legend(
    handles=legend_elements,
    loc='upper left',
    ncol=1,
    columnspacing=1,
    handletextpad=0.5,
    fontsize=25  # 设置图例的字体大小
)

# 设置标题和坐标轴标签并调整字体大小
# ax.set_title('Paper_c1 All Fields', fontsize=30)  # 设置标题字体大小
ax.set_xlabel('Years after publication', fontsize=30)  # 设置x轴标签字体大小
ax.set_ylabel('Harm value (%)', fontsize=30)  # 设置y轴标签字体大小

# 设置坐标轴范围和刻度格式
ax.set_ylim(top=300)  # 修改纵坐标上限
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
ax.tick_params(axis='both', which='major', labelsize=30)  # 修改主要刻度的字体大小
ax.tick_params(axis='both', which='minor', labelsize=30)

# 创建插入图
inset_ax = inset_axes(ax, width="40%", height="40%", loc='upper right',
                      bbox_to_anchor=(-0.18, -0.13, 1.13, 1.13), bbox_transform=ax.transAxes)  # 向上移动插入图
inset_ax.plot(positions, q2, marker='^', color='#8B4513', markersize=13, linestyle='--', linewidth=3)
inset_ax.spines['top'].set_visible(False)
inset_ax.spines['right'].set_visible(False)
# 动态调整插入图纵坐标范围
padding = 5  # 设置额外的上下留白
inset_ax.set_xlim(1, len(q2) )  # 限制x轴范围
# inset_ax.set_ylim(min(q2) - padding, max(q2) + padding)  # 动态调整y轴范围
inset_ax.set_ylim(min(q2) - padding, 50)  # 动态调整y轴范围

inset_ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
inset_ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
inset_ax.tick_params(axis='both', which='major', labelsize=20)
# 添加连接线
from matplotlib.patches import ConnectionPatch
for i, pos in enumerate(positions):
    if i == 0 or i == len(positions) - 1:  # 连接首尾点
        # con = ConnectionPatch(xyA=(pos, q2[i]), xyB=(pos, q2[i]), coordsA="data", coordsB="data",
        #                       axesA=ax, axesB=inset_ax, color="black", lw=1)
        con = ConnectionPatch(
            xyA=(pos, q2[i]), xyB=(pos, q2[i]), coordsA="data", coordsB="data",
            axesA=ax, axesB=inset_ax, color="black", lw=2, linestyle=(0, (10, 5))  # 10 像素线段，5 像素间距
        )
        fig.add_artist(con)

# 保存图像
plt.savefig('fig2_a_final.pdf', format='pdf', bbox_inches='tight')
