# %%
import numpy as np
import scipy.signal as signal
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

cof_type_all = ['plv','cos']
cof_type = cof_type_all[0]
corr_type_all = ['pearson_r','cos_sim']
corr_type = corr_type_all[1]

# %%
# 功能：计算nFC矩阵
# 输入：原始数据希尔伯特变换后的相位数据
# 输出：nFC矩阵
def nFC_corr(ts):
    # T，N，M=时间，通道数/节点数，边数
    Phase = np.angle(signal.hilbert(ts))
    T,N = Phase.shape

    # 边瞬时强度的时间序列，维度为时间*边数
    if corr_type=='pearson_r':
        nFC = np.corrcoef(Phase.T)
    elif corr_type=='cos_sim':
        nFC = cos_sim(Phase.T,Phase.T)

    return nFC

# %%
# 功能：计算nFC矩阵
# 输入：原始数据希尔伯特变换后的相位数据
# 输出：nFC矩阵
def nFC_exp(ts):
    # T，N，M=时间，通道数/节点数，边数
    Phase = np.angle(signal.hilbert(ts))
    T,N = Phase.shape

    # 边瞬时强度的时间序列，维度为时间*边数
    nFC = np.zeros([N,N])
    
    for i in range(N):
        for j in range(N):
            # 进行希尔伯特变换,得到每个时间点𝑡的瞬时相位信息
            # nFC矩阵
            phi_dlt = Phase[:,i]-Phase[:,j]
            nFC_cos = np.sum(np.cos(phi_dlt))
            nFC_sin = np.sum(np.sin(phi_dlt))
           
           # plv形式
            if cof_type=='plv':
                nFC[i,j] = np.sqrt(nFC_cos**2+nFC_sin**2)/T
            elif cof_type=='cos':
                nFC[i,j] = nFC_cos/T
    
    return nFC

def nFC_delta(ts):
    # T，N，M=时间，通道数/节点数，边数
    Phase = np.angle(signal.hilbert(ts))
    T,N = Phase.shape

    # 边瞬时强度的时间序列，维度为时间*边数
    nFC = np.zeros([N,N])
    
    for i in range(N):
        for j in range(N):
            # 进行希尔伯特变换,得到每个时间点𝑡的瞬时相位信息
            # 计算两个结点相位信息的z-score，得到nFC矩阵
            # 相位差的绝对值求平均
            nFC[i,j] = np.mean(abs(Phase[:,i]-Phase[:,j]))
    
    return nFC

# %%
# nFC_mat测试代码
'''import scipy.io as scio
import scipy.signal as signal

path_mat='data/Class1-Parent/2021060701epoch(1-3-6-9-12)_noref.mat'
data_mat = scio.loadmat(path_mat)
# data_mat的格式为情况*1*频段
# 选择目标模态对应的数据，ts的格式为被试数*时间点数*通道数
ts = data_mat['ca23'][0][1].T
nFC = nFC_zsc(ts[0])
print(nFC)'''


