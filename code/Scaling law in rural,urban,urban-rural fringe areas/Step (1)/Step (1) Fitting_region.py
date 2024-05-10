import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from matplotlib import rcParams
from sklearn.metrics import mean_absolute_error, mean_squared_error


rcParams['font.serif'] = ['Times New Roman']

data = pd.read_csv('Regional_travel_population.csv', encoding="GB18030")
#It is necessary to enter the three parameters of transition, rural, and Urban respectively to obtain the fitting results of these three regions.
# Distinguish area
urban = data[(data['code']==1)]
transition = data[data['code']==2]
rural = data[data['code']==3]


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
    #data_shuru2 = data_shuru[data_shuru['renkou_quxian']>10**5.5]
    x1 = np.log10(data_shuru['population_district'])
    y1 = np.log10(data_shuru['total_frequency'])
    a = x1.to_numpy(dtype='float32')
    b = y1.tolist()

    x2 = np.log10(data_shuru['population_district'])
    y2 = np.log10(data_shuru['total_distance'])
    c = x2.to_numpy(dtype='float32')
    d = y2.tolist()
    return a, b, c,d

# Input
X_renkou,Y_count,X_renkou2,Y_sumdistance = shuru(transition)# Need to change parameters!!!
p0_count = [np.mean(X_renkou), np.mean(Y_count), 1, 1]
p0_sumdistance = [np.mean(X_renkou2), np.mean(Y_sumdistance), 1, 1]

# Perform fitting
params_count, covariance_count = curve_fit(piecewise_linear, X_renkou, Y_count, p0_count)
params_sumdistance, covariance_sumdistance = curve_fit(piecewise_linear, X_renkou2, Y_sumdistance, p0_sumdistance)

# Extract parameters

x_break_count, y_break_count, k1_count, k2_count = params_count
x_break_sumdistance, y_break_sumdistance, k1_sumdistance, k2_sumdistance = params_sumdistance

print('total_frequency:',x_break_count, y_break_count,k1_count,k2_count)
print('total_distance',x_break_sumdistance, y_break_sumdistance,k1_sumdistance,k2_sumdistance)

X_line = np.arange(min(X_renkou), max(X_renkou), 0.01)
Y_line_count = piecewise_linear(X_line, x_break_count, y_break_count, k1_count, k2_count)
Y_line_sumdistance = piecewise_linear(X_line, x_break_sumdistance, y_break_sumdistance, k1_sumdistance, k2_sumdistance)

# Calculate fitting accuracy
r2_count, mae_count, rmse_count = fit_evaluation(X_renkou,Y_count,x_break_count, y_break_count, k1_count, k2_count)
r2_sumdistance, mae_sumdistance, rmse_sumdistance = fit_evaluation(X_renkou,Y_sumdistance,x_break_sumdistance, y_break_sumdistance, k1_sumdistance, k2_sumdistance)
print('total_frequency_evaluation',r2_count, mae_count, rmse_count )
print('total_distance_evaluation',r2_sumdistance, mae_sumdistance, rmse_sumdistance)

# Create DataFrame
df_params = pd.DataFrame({
    'Parameter': ['x_break_frequency', 'y_break_frequency', 'k1_frequency', 'k2_frequency', 'x_break_sumdistance', 'y_break_sumdistance', 'k1_sumdistance', 'k2_sumdistance'],
    'Value': [x_break_count, y_break_count, k1_count, k2_count, x_break_sumdistance, y_break_sumdistance, k1_sumdistance, k2_sumdistance]
})

# Create DataFrame for fitting accuracy metrics
df_accuracy = pd.DataFrame({
    'Metric': ['R-squared', 'MAE', 'RMSE'],
    'total_frequency': [r2_count, mae_count, rmse_count],
    'total_distance': [r2_sumdistance, mae_sumdistance, rmse_sumdistance]
})

# Save DataFrames to CSV files
df_params.to_csv('urban-rural fringe_fitting parameters.csv', index=False)
df_accuracy.to_csv('urban-rural fringe_fitting accuracy.csv', index=False)


