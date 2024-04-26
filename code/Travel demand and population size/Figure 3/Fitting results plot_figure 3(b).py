import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from matplotlib import rcParams
from sklearn.metrics import mean_absolute_error, mean_squared_error


rcParams['font.serif'] = ['Times New Roman']

# Input
data = pd.read_csv('District_travel_population.csv', encoding="GB18030")
data_sandian = pd.read_csv(r'Individual distribution fitting results.csv', encoding="GB18030")

# Define piecewise linear functions
def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0, x >= x0], [lambda x: k1 * (x - x0) + y0, lambda x: k2 * (x - x0) + y0])

# Calculate goodness of fit
def fit_evaluation(X_shuru,Y_shuru, x_break,y_break,k1,k2):
    residuals = Y_shuru - piecewise_linear(X_shuru, x_break,y_break,k1,k2)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((Y_shuru - np.mean(Y_shuru)) ** 2)
    r2 = 1 - (ss_res / ss_tot)  # 拟合优度
    mae = mean_absolute_error(Y_shuru, piecewise_linear(X_shuru, x_break,y_break,k1,k2))
    rmse = np.sqrt(mean_squared_error(Y_shuru, piecewise_linear(X_shuru, x_break,y_break,k1,k2)))
    return r2, mae, rmse
# Input
def shuru(data_shuru):
    x1 = np.log10(data_shuru['population_district'])
    y1 = np.log10(data_shuru['total_frequency'])
    a = x1.to_numpy(dtype='float32')
    b = y1.tolist()
    return a, b,


plt.rcParams["font.family"] = "Times New Roman"
def plot(x_fit_renkou, y_fit_count, highlight_point1, sandian):
    fig, ax1 = plt.subplots(figsize=(8, 6),dpi=100)

    # Plot the line graph with log y-axis on the left
    ax1.plot(10 ** x_fit_renkou, 10 ** y_fit_count, color='#ff6b35', label='Aggregation frequency fitting result')
    ax1.set_yscale('log')
    ax1.set_ylabel('Aggregation frequency', fontsize=16, fontweight='bold')
    ax1.set_ylim(10 **2, 10 ** 7)

    # Highlight the specified point
    if highlight_point1:
        ax1.scatter(highlight_point1[0], highlight_point1[1], color='#ae2012', s=100, marker='^', label='Turning point')

    # Plot the scatter plot with linear y-axis on the right
    ax2 = ax1.twinx()
    ax2.scatter(sandian['population_district'], sandian['k'].abs(), color='#1a659e', label='k')
    ax2.set_ylabel('Slope of individual frequency distribution ', fontsize=16, fontweight='bold')  # Update with your desired label
    ax2.set_ylim(2.8, 7.3)  # Update with your desired limits for the scatter plot

    # Draw a vertical line at x = 10^5.5
    ax1.axvline(x=10 ** 5.8, color='#588157', linestyle='--',linewidth=1 )

    # Draw a vertical line passing through highlight_point1
    x_point, y_point = highlight_point1
    ax1.axvline(x=x_point, color='#588157', linestyle='--', linewidth=1)
    ax1.set_xscale('log')
    ax1.set_xlim(10 ** 5, 10 ** 7)
    # Set titles and legends
    ax1.set_xlabel('Population size', fontsize=16, fontweight='bold')
    ax1.grid(which='major', axis='y')
    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.legend(loc='upper left',fontsize=14, title='Parameters',title_fontsize=14)
    ax2.legend(loc='upper right',fontsize=14, title='Parameters',title_fontsize=14)
    ax2.tick_params(axis='y', labelsize=12)
    plt.title('(b)', loc='left', fontsize=24)
    plt.savefig('Figure 3(b)')
    plt.show()


# Input
X_renkou,Y_count = shuru(data)
p0_count = [np.mean(X_renkou), np.mean(Y_count), 1, 1]

# Fitting
params_count, covariance_count = curve_fit(piecewise_linear, X_renkou, Y_count, p0_count)

# Extract parameters
x_break_count, y_break_count, k1_count, k2_count = params_count
print('total_frequency:',x_break_count, y_break_count,k1_count,k2_count)

X_line = np.arange(min(X_renkou), max(X_renkou), 0.01)
Y_line_count = piecewise_linear(X_line, x_break_count, y_break_count, k1_count, k2_count)


# Calculate fitting accuracy
r2_count, mae_count, rmse_count = fit_evaluation(X_renkou,Y_count,x_break_count, y_break_count, k1_count, k2_count)
print('total_frequency_evaluation',r2_count, mae_count, rmse_count )

# Picture
highlighted_point_count = (10**x_break_count,10**y_break_count)
plot(X_line,Y_line_count,highlighted_point_count,data_sandian)



