import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Data
data = pd.read_csv(r'District_distance_travel_population.csv', encoding="GB18030")
data_range = data[data['distance_range'] == '[18469-35640]'] # Need to change parameters: distance range, and enter the distance group separately!!!

# Define piecewise linear functions
def piecewise_linear(params, x):
    x0, y0, k1, k2 = params
    return np.piecewise(x, [x < x0, x >= x0], [lambda x: k1 * (x - x0) + y0, lambda x: k2 * (x - x0) + y0])

# Define the loss function
def loss_function(params, x, y):
    y_pred = piecewise_linear(params, x)
    return np.sum((y_pred - y)**2)

# Input
def process_input(data):
    x = np.log10(data['population_district'].values.astype(np.float32))
    y = np.log10(data['total_frequency'].values.astype(np.float32))
    return x, y

# Calculate fit evaluation index
def evaluate_fit(x, y, params):
    y_pred = piecewise_linear(params, x)
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    return r2, mae, rmse

# Picture
def plot_fit(x_true, y_true, params,break_x,break_y):
    x_fit = np.linspace(min(x_true), max(x_true), 1000)
    y_fit = piecewise_linear(params, x_fit)
    plt.scatter(10**x_true, 10**y_true, color='lightcoral', marker='o', label='Frequency')
    plt.scatter(10**break_x, 10**break_y, color='blue', marker='^', label='break')
    plt.plot(10**x_fit, 10**y_fit, color='red', label='Frequency_Fit')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Group9', fontsize=16, fontweight='bold')
    plt.xlabel('Population_District', fontsize=14, fontweight='bold')
    plt.ylabel('Frequency', fontsize=14, fontweight='bold')
    plt.ylim(10**2, 10**8)
    plt.xlim(10**5, 10**7)
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True)
    plt.savefig('Group9')
    plt.show()

# Inout
x_data, y_data = process_input(data_range)
initial_guess = [np.mean(x_data), np.mean(y_data), 1.32345, 0.99]

# Fitting
result = minimize(loss_function, initial_guess, args=(x_data, y_data), method='Powell')

# Extract fitting results and parameters:get the fitting parameters for each distance group
params_fit = result.x
x_break_count, y_break_count, k1_count, k2_count= params_fit
print('Fitted Parameters:', params_fit)

#  Picture
plot_fit(x_data, y_data, params_fit,x_break_count, y_break_count)
# Calculate fit evaluation index
r2_fit, mae_fit, rmse_fit = evaluate_fit(x_data, y_data, params_fit)
print('Fit Evaluation:', r2_fit, mae_fit, rmse_fit)
