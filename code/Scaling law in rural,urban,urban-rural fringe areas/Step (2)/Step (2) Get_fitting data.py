
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# read data
df = pd.read_csv('Population_district.csv',encoding="GB18030")
df['log_population_district'] = np.log10(df['population_district'])

# Perform interpolation
sorted_population = np.sort(df['log_population_district'].unique())
new_x = np.linspace(sorted_population.min(), sorted_population.max(), 100)
f = interp1d(sorted_population, sorted_population, kind='linear')
data = pd.DataFrame({'log_population_district': new_x})
data['population_district'] = np.power(10, data['log_population_district'])

# Piecewise function
def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0, x >= x0], [lambda x: k1 * (x - x0) + y0, lambda x: k2 * (x - x0) + y0])

# Calculation based on step1 parameters:parameters need to be replaced according to step（1）
data['log_total_frequency'] = data['log_population_district'].apply(lambda x: piecewise_linear(x, 6.320189273, 5.073288755, 0.853840135, 3.281109688))
data['log_total_distance'] = data['log_population_district'].apply(lambda x: piecewise_linear(x,6.215468601, 6.08404669,0.686032568, 2.523680479))
data['total_frequency']= np.power(10,data['log_total_frequency'])
data['total_distance']= np.power(10,data['log_total_distance'])
data['code']=3

# Output
data.to_csv('urban-rural fringe_fitting results.csv',encoding="GB18030")
print(data[['population_district', 'log_population_district', 'log_total_frequency','total_frequency','total_distance', 'log_total_distance']])


