import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from matplotlib import rcParams
from sklearn.metrics import mean_absolute_error, mean_squared_error


#Read data
data = pd.read_csv('Patterns_travel_population.csv', encoding="GB18030")
data_commuting = data[data['travel_pattern']==1]
data_leisure = data[data['travel_pattern']==2]


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

# Input control
def shuru(data_shuru):
    #data_shuru2 = data_shuru[data_shuru['renkou_quxian']>10**5.5]
    x1 = np.log10(data_shuru['renkou_quxian'])
    y1 = np.log10(data_shuru['count'])
    a = x1.to_numpy(dtype='float32')
    b = y1.tolist()

    x2 = np.log10(data_shuru['renkou_quxian'])
    y2 = np.log10(data_shuru['sum_distance'])
    c = x2.to_numpy(dtype='float32')
    d = y2.tolist()
    return a, b, c,d,

plt.rcParams["font.family"] = "Times New Roman"
def plot(x_true_renkou,y_true_count,y_true_sumdistance, x_fit_renkou,y_fit_count,y_fit_sumdistance,
         highlight_point1,highlight_point2):
    rcParams['font.serif'] = ['Times New Roman']
    plt.scatter(x_true_renkou, y_true_count, color='lightcoral', marker='o', label='Frequency')
    plt.plot(10 ** x_fit_renkou, 10 ** y_fit_count, color='red', label='Fitting result')
    if highlight_point1:
        # Highlight the specified point
        plt.scatter(highlight_point1[0], highlight_point1[1], color='red', s=100, marker='^', label='Turning point')


    plt.scatter(x_true_renkou, y_true_sumdistance, color='lightsteelblue', marker='o', label='Distance')
    plt.plot(10 ** x_fit_renkou, 10 ** y_fit_sumdistance, color='blue', label='Fitting result')
    if highlight_point2:
        # Highlight the specified point
        plt.scatter(highlight_point2[0], highlight_point2[1], color='blue', s=100, marker='^', label='Turning point')

    plt.xscale('log')
    plt.yscale('log')
    plt.title('(a)Commuting', fontsize=20,fontweight='bold',loc='left')
    plt.xlabel('Population size', fontsize=14, fontweight='bold')
    plt.ylabel('Travel demand', fontsize=14, fontweight='bold')
    plt.ylim(10**2,10**8)
    plt.xlim(10**5,10**7)
    plt.legend(fontsize=10,loc='lower right')
    plt.grid(axis='y')
    plt.subplots_adjust(top=0.92, bottom=0.13)
    plt.savefig('Commuting3')
    plt.show()


# Input
X_renkou,Y_count,X_renkou2,Y_sumdistance, = shuru(data_commuting)# Need to replace parameters: data_commuting, data_leisure
p0_count = [np.mean(X_renkou), np.mean(Y_count), 1, 1]
p0_sumdistance = [np.mean(X_renkou2), np.mean(Y_sumdistance), 1, 1]

# Perform fitting
params_count, covariance_count = curve_fit(piecewise_linear, X_renkou, Y_count, p0_count)
params_sumdistance, covariance_sumdistance = curve_fit(piecewise_linear, X_renkou2, Y_sumdistance, p0_sumdistance)

# Extract parameters
x_break_count, y_break_count, k1_count, k2_count = params_count
x_break_sumdistance, y_break_sumdistance, k1_sumdistance, k2_sumdistance = params_sumdistance
print('count:',x_break_count, y_break_count, k1_count, k2_count)
print('sum_distance',x_break_sumdistance, y_break_sumdistance, k1_sumdistance, k2_sumdistance)


# Calculate
X_line = np.arange(min(X_renkou), max(X_renkou), 0.01)
Y_line_count = piecewise_linear(X_line, x_break_count, y_break_count, k1_count, k2_count)
Y_line_sumdistance = piecewise_linear(X_line, x_break_sumdistance, y_break_sumdistance, k1_sumdistance, k2_sumdistance)


# Calculate goodness of fit
r2_count, mae_count, rmse_count = fit_evaluation(X_renkou,Y_count,x_break_count, y_break_count, k1_count, k2_count)
r2_sumdistance, mae_sumdistance, rmse_sumdistance = fit_evaluation(X_renkou,Y_sumdistance,x_break_sumdistance, y_break_sumdistance, k1_sumdistance, k2_sumdistance)
print('count_evaluation',r2_count, mae_count, rmse_count )
print('sumdistance_evaluation',r2_sumdistance, mae_sumdistance, rmse_sumdistance)

highlighted_point_count = (10**x_break_count,10**y_break_count)
highlighted_point_sumdistance = (10**x_break_sumdistance, 10**y_break_sumdistance)
plot(data_commuting['renkou_quxian'],data_commuting['count'],data_commuting['sum_distance'],
     X_line,Y_line_count,Y_line_sumdistance,
     highlighted_point_count,highlighted_point_sumdistance)



