import os
import glob
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D


data_csvs = glob.glob(f'./1normal/normal-D/paper_c[2-6]/all-*.csv')

plt.rcParams.update({
    'font.family': 'Times New Roman',   # 需要系统支持这个字体，没有可以注释
    'font.weight': 'semibold',
    'axes.labelweight': 'semibold',
    'axes.titleweight': 'semibold',
})

fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)  # 2行3列，调整画布大小，不共享x轴
axes = axes.flatten()

labels = ['a', 'b', 'c', 'd', 'e']


def get_color(x: float) -> str:
    if x < 0.0:
        return 'blue'
    elif x > 0.75:
        return 'red'
    else:
        return 'orange'


for i, ax in enumerate(axes):
    data = np.loadtxt(data_csvs[i], delimiter=",")
    paper_c = f"Paper_c{i + 1}"
    q3 = data[0]
    q2 = data[1]
    q1 = data[2]

    positions = np.arange(1, len(q2) + 1)

    for pos, lower, upper, mid_v in zip(positions, q1, q3, q2):
        # pos 是位置，lower 和 upper 是边界值
        segments = [
            ((-np.inf, 0), 'blue'),
            ((0, 0.75), 'orange'),
            ((0.75, np.inf), 'red')
        ]

        for (segment_min, segment_max), color in segments:
            # 将数据裁剪到当前段的范围内
            y_bottom = np.clip(lower, segment_min, segment_max)
            y_top = np.clip(upper, segment_min, segment_max)

            # 只画有高度差的线段
            mask = y_top > y_bottom
            if np.any(mask):
                ax.vlines(pos[mask], y_bottom[mask], y_top[mask], colors=color, linewidth=2)

        # 绘制 Q1 和 Q3 的横线，形成工字型的上下横线
        ax.hlines(lower, pos - 0.15, pos + 0.15, colors=get_color(lower), linewidth=2)
        ax.hlines(upper, pos - 0.15, pos + 0.15, colors=get_color(upper), linewidth=2)

    ax.plot(positions, q2, marker='o', color='orange', markersize=10)
    if i == 0:
        legend_elements = [
            Line2D([0], [0], color='blue', label='Harm < 0.0'),
            Line2D([0], [0], color='orange', label='0.0 ≤ Harm ≤ 0.75'),
            Line2D([0], [0], color='red', label='Harm > 0.75'),
            Line2D([0], [0], color='orange', marker='o', label='Median')
        ]
        ax.legend(
            handles=legend_elements,
            loc='upper left',  # 位置在左上角
            ncol=2,  # 设置为2列
            columnspacing=1,  # 列间距
            handletextpad=0.5  # 图例符号和文本之间的间距
        )

    ax.set_title(f'{paper_c} all fields')

    if i >= 3:
        ax.set_xlabel('Year after publication')
    if i % 3 == 0:
        ax.set_ylabel('Harm value')

    ax.set_ylim(top=1.5)

    # 添加子图标签
    ax.text(-0.13, 1.1, labels[i], transform=ax.transAxes,
            fontsize=16, va='top')

    # 设置x,y轴格式
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))  # 每1个单位一个刻度
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

# plt.tight_layout()
plt.savefig('paper_c_all.pdf', format='pdf', bbox_inches='tight')
# plt.show()