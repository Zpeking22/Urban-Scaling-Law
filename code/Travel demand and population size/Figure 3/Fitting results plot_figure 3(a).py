import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Individual distribution fitting results
data = pd.read_csv(r'Individual distribution fitting results.csv', encoding="GB18030")
data = data[['district', 'k', 'b', 'population_district']]

# Gradient color based on population size
colors = np.array(data['population_district'])
colors_shiyong = (colors - min(colors)) / (max(colors) - min(colors))

# Picture
plt.rcParams["font.family"] = "Times New Roman"  # 设置字体为新罗马字体
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(len(data['district'])):
    x = np.linspace(0, 2, 100)
    y = data['k'][i] * x + data['b'][i]
    x_true = 10**x
    y_true = 10**y
    ax.plot(x_true, y_true, color=plt.cm.plasma(colors_shiyong[i]), alpha=0.8)

scatter = ax.scatter([], [], c=[], cmap='plasma', label='Population')
handles, _ = scatter.legend_elements(prop="sizes", num=3)

# Set color bar
cbar = plt.colorbar(scatter)
cbar.set_label('Population size',fontsize=16)
# Set colorbar ticks and ticklabels
tick_labels = ["{:.1e}".format(value) for value in [min(colors), np.percentile(colors, 25), np.percentile(colors, 50), np.percentile(colors, 75), max(colors)]]
cbar.set_ticklabels(tick_labels)
ax.set_title('(a)', loc='left', fontsize=24)
ax.set_xlabel('Individual frequency',fontsize=16, fontweight='bold')
ax.set_ylabel('P(Individual frequency)',fontsize=16, fontweight='bold')
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(10**-7)
plt.savefig('Figure 3(a)')
plt.show()



