import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.family"] = "Times New Roman"  # 设置字体为新罗马字体

# Picture
def plot_segments(data):
    fig = plt.figure(figsize=(8, 6))
    for i, segment in enumerate(data):
        distance_range, break_x, break_y, k1, k2, population = segment

        x_range_k1 = np.linspace(3.5, break_x, 100)
        x_range_k2 = np.linspace(break_x, 8, 100)


        y_k1 = k1 * (x_range_k1 - break_x) + break_y
        y_k2 = k2 * (x_range_k2 - break_x) + break_y

        # color
        color = i / len(data)
        alpha = i / len(data)

        plt.plot(10 ** x_range_k1, 10 ** y_k1, color=plt.cm.plasma(color), label=f'{distance_range}')
        plt.plot(10 ** x_range_k2, 10 ** y_k2, color=plt.cm.plasma(color))
        plt.scatter(10 ** break_x, 10 ** break_y, color=plt.cm.plasma(color), marker='^')  # 标记转折点

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Population size', fontsize=16, fontweight='bold')
    plt.ylabel('Frequency', fontsize=16, fontweight='bold')
    legend = plt.legend(loc='lower right',fontsize=14, title='Distance(km)',title_fontsize=14)
    legend.set_title('Distance(km)', prop={'size': 14})
    plt.subplots_adjust(top=0.9, bottom=0.17)  # 调整底部空白区域的大小
    plt.title('(a)', loc='left', fontsize=24)
    plt.setp(legend.get_texts(), fontsize=14)  # 设置图例中的文字大小为14号
    plt.savefig('Figure 2(a)')
    plt.show()


# The fitting parameters of each distance group that have been obtained
data = [
    ['(0, 0.79]', 6.21662065, 4.89034836, 1.73926254, 0.84930946, 100000],
    ['(0.79, 1.77]', 6.21547585, 5.0813989, 1.85590917, 0.87650594, 200000],
    ['(1.77, 2.50]', 6.22007438, 4.93562419, 1.77527856, 0.97831305, 50000],
    ['(2.50, 3.61]', 6.23736317, 5.00069007, 1.75377108, 0.92873243, 150000],
    ['(3.61, 5.26]', 6.2542708, 5.00192214, 1.64906311, 0.83903386, 300000],
    ['(5.26, 7.46]', 6.2598765, 5.02033426, 1.67083675, 0.75478944, 40000],
    ['(7.46, 11.18]', 6.35060008, 5.1379853, 1.63252365, 0.56313682, 250000],
    ['(11.18, 18.47]', 6.38309109, 5.16683287, 1.66261619, 0.41721991, 1000000],
    ['(18.47, 35.64]', 6.2397024, 4.90223065, 1.56282696, 0.84048282, 80000],
    ['(35.64,   max]', 6.32026124, 4.83611311, 1.05746281, 0.34704368, 120000]
]

# Picture
plot_segments(data)



