import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #绘制回归曲线
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Due to personal privacy protection, individual travel data cannot be provided
data = pd.read_csv(r'Individual travel.csv',encoding="GB18030")

# Logarithmic binning
def log_nbinning(data):
    bins = np.logspace(0, 1.5, 10)
    widths = (bins[1:] - bins[:-1])
    total_sums_list = [0] * (len(bins) - 1)
    for index, row in data.iterrows():
        value = row['frequency']
        frequency_distribution = row['frequency_distribution']
        for i in range(len(bins) - 1):
            start, end = bins[i], bins[i + 1]
            if start <= value < end:
                total_sums_list[i] += frequency_distribution
                break
    # normalize by bin width
    hist_norm = total_sums_list/widths
    zong = np.sum(hist_norm)
    hist_norm2 = hist_norm/zong
    return bins[:-1], hist_norm2

# Input processing
def shuru_chuli(x,y):
    bine = pd.DataFrame({'x': x, 'y': y})
    bine = bine[bine['y'] > 0]
    bine['a'] = np.log10(bine['x'])
    bine['b'] = np.log10(bine['y'])
    a = bine['a'].to_numpy(dtype ='float32')
    b = bine['b'].tolist()
    return a,b

# Regression
def regression(x,y):
    reg = LinearRegression()
    model = reg.fit(x, y)
    Y = model.predict(x)
    k = model.coef_
    b = model.intercept_
    r2 = model.score(x, y)
    rmse = np.sqrt(mean_squared_error(y, Y))
    mae = mean_absolute_error(y, Y)
    return k, b, x, Y,r2,rmse,mae

# Picture
def plot_all(x_0,y_0,xl_0,yl_0,):
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.subplots_adjust(left=None, bottom=0.15, right=None, top=None, wspace=None, hspace=0.5)
    scatter1 = plt.scatter(x_0, y_0, c='r', marker='o', label='cluster1')
    line0 = plt.plot(10 ** xl_0, 10 ** yl_0, c='r',linewidth=1,linestyle='--')
    plt.legend(loc='lower left')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Frequency', fontsize=15)
    plt.ylabel('P(Frequency)', fontsize=15)
    plt.title("shishi", fontsize=16)
    plt.savefig('shishi')
    plt.show()

#quxian=['罗湖区','福田区','南山区','宝安区','龙岗区','盐田区','龙华区','坪山区']
#quxian=['禅城区','南海区','顺德区','三水区','高明区']
#quxian=['惠城区','惠阳区','博罗县','惠东县','龙门县']
#quxian =['蓬江区','江海区','新会区','台山市','开平市','鹤山市','恩平市']
#quxian = ['端州区','鼎湖区','高要区','广宁县','怀集县','封开县','德庆县','四会市']
#quxian = ['香洲区','斗门区','金湾区']
quxian=['荔湾区','越秀区','海珠区','天河区','白云区','黄埔区','番禺区','花都区','南沙区','从化区','增城区']
quxian_k = pd.DataFrame(columns=['name', 'k', 'b', 'r2', 'rmse', 'mae'])
for name in quxian:
    data_quxian = data[data['zone_name'] == name]
    x0, y0 = log_nbinning(data_quxian)
    xj0, yj0 = shuru_chuli(x0, y0)
    k, b, X0, Y0, r2, rmse, mae = regression(xj0.reshape(-1, 1), yj0)
    quxian_k = quxian_k.append({
        'name': name,
        'k': k[0],
        'b': b,
        'r2': r2,
        'rmse': rmse,
        'mae': mae
    }, ignore_index=True)
    print(f'{name}: k={k[0]}, b={b}, r2={r2}, rmse={rmse}, mae={mae}')

# Output
quxian_renkou = pd.read_csv(r'quxian_count_renkou.csv',encoding="GB18030")
quxian_renkou = quxian_renkou.rename(columns={'区县':'name'})
result_df = pd.merge(quxian_k, quxian_renkou,on='name', how='left')
result_df.to_csv('Individual distribution fitting results.csv', index=False,encoding="GB18030")
# ‘Individual distribution fitting results.csv' has been provided
