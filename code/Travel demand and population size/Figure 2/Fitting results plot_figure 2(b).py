import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Times New Roman"

# The fitting parameters of each distance group
data = pd.read_csv(r'Fitting results of each distance group.csv', encoding="GB18030")

# Picture
def plot_segments(data):
    fig, ax1 = plt.subplots(figsize=(8, 6))  # 创建图形和左侧坐标轴
    ax2 = ax1.twinx()  # 创建右侧坐标轴
    ax1.plot(data.distance, data.k1, color='#f94144', label='β1')
    ax1.plot(data.distance, data.k2, color='#f3722c', label='β2')
    ax1.scatter(data.distance, data.k1, color='#f94144', marker='o')
    ax1.scatter(data.distance, data.k2, color='#f3722c', marker='o')
    ax2.scatter(data.distance, data.break_x, color='#277da1', marker='^')
    ax2.plot(data.distance, data.break_x, color='#277da1', linestyle='--', label='Turning point')
    ax1.set_xlabel('Distance(km)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Slopes within two stages ', fontsize=16, fontweight='bold')
    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.set_ylim([0.3,3])
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30)
    ax1.xaxis.set_label_coords(0.5, -0.16)  # 调整 x 轴标签的位置,上下左右
    ax2.set_ylabel('Turning point', fontsize=16, fontweight='bold')
    ax2.set_ylim([5.9, 6.4])
    ax2.tick_params(axis='y', labelsize=12)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax2.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=14, title='Parameters',title_fontsize=14)
    plt.title('(b)',loc='left',fontsize=24)
    plt.subplots_adjust(top=0.9,bottom=0.17)  # 调整底部空白区域的大小
    plt.grid()
    plt.savefig('Figure 2(b).png')  # 保存图片为 k_dual_axes.png
    plt.show()

# Picture
plot_segments(data)
