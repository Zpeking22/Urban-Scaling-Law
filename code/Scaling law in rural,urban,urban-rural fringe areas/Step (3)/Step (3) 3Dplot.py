import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from matplotlib import rcParams
import matplotlib as mpl

# read data
df = pd.read_csv('Fitting data.csv', encoding="GB18030")

# Need to change parameters: distance, frequency
s = 'Distance'
# Create a pivot table to organize data
pivot_table = df.pivot_table(index='code', columns='log_population_district', values='log_total_distance')

# Get the values of X, Y, Z
X = pivot_table.columns.values
Y = pivot_table.index.values
X, Y = np.meshgrid(X, Y)
Z = pivot_table.values

# Create 3D plot
mpl.rcParams['font.family'] = 'Times New Roman'
fig = plt.figure(dpi=200)
ax = fig.add_subplot(111, projection='3d')

# Plot 3D wireframe
#ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
ax.plot_surface(X, Y, Z, rstride=4, cstride=4, cmap='viridis')
plt.title('(b)Aggregate distance in different regions', fontsize=20, fontweight='bold', loc='left')
plt.subplots_adjust(top=0.92, bottom=0.05, left=0.01)

# Plot line for y value 1
y_1_indices = np.where(Y == 1)
y_1_x = X[y_1_indices]
y_1_z = Z[y_1_indices]
ax.plot(y_1_x, np.ones_like(y_1_x), y_1_z, color='#007f5f', zorder=3, label='Rural area')

# Plot line for y value 2
y_2_indices = np.where(Y == 2)
y_2_x = X[y_2_indices]
y_2_z = Z[y_2_indices]
ax.plot(y_2_x, np.ones_like(y_2_x) * 2, y_2_z, color='#f35b04', zorder=3, label='Urban-rural fringe')

# Plot line for y value 3
y_3_indices = np.where(Y == 3)
y_3_x = X[y_3_indices]
y_3_z = Z[y_3_indices]
ax.plot(y_3_x, np.ones_like(y_3_x) * 3, y_3_z, color='red', label='Urban area')
ax.legend(fontsize=10, loc='upper right')

# Set axis labels
ax.set_yticks(np.arange(1, 4, 1))
ax.set_xlabel('Population size', size=14)
ax.set_ylabel('Region', size=14, labelpad=20)
ax.set_zlabel(s, size=14)

# Get the text content of x-axis tick labels and convert to float
plt.xticks(fontname='Times New Roman')
xtick_labels_float = [float(label.get_text()) for label in ax.get_xticklabels()]
xtick_labels_log = ['$10^{{{}}}$'.format(x) for x in xtick_labels_float]
ax.set_xticklabels(xtick_labels_log, fontname='Times New Roman')
ax.tick_params(axis='x', labelsize=8)

# Change y-axis ticks
new_labels = ['rural', 'urban-rural fringe', 'urban']
ax.set_yticklabels(new_labels, fontname='Times New Roman')
ax.tick_params(axis='y', pad=3, labelsize=10)

# Get the text content of z-axis tick labels and convert to float
ztick_labels_float = [float(label.get_text()) for label in ax.get_zticklabels()]
ztick_labels_log = ['$10^{{{}}}$'.format(x) for x in ztick_labels_float]
ax.set_zticklabels(ztick_labels_log, fontname='Times New Roman')
ax.tick_params(axis='z', labelsize=8)

plt.savefig('Total ' + s + '.jpg')
plt.show()
