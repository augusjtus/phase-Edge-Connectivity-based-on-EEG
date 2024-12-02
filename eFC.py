# %%
import numpy as np
import scipy.signal as signal
from scipy.stats import zscore
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

'''cof_type_all = ['real','cos','sin']
cof_type = cof_type_all[1]'''

# %%
# 功能：计算边瞬时强度的时间序列
# 输入：原始数据希尔伯特变换后的相位数据
# 输出：边瞬时强度的时间序列
def Int_exp(ts):
    # T，N，M=时间，通道数/节点数，边数
    Phase = np.angle(signal.hilbert(ts))
    T,N = Phase.shape
    M = int(N*(N-1)/2)

    # 边瞬时强度的时间序列，维度为时间*边数
    I_edge = np.zeros([T,M])
    e_num = 0
    edges = []
    nodes1 = []
    nodes2 = []
    
    for i in range(1,N):
        for k in range(i):
            # 进行希尔伯特变换,得到每个时间点𝑡的瞬时相位信息
            # 计算两点相位差的cos函数，得到边的瞬时强度
            phi_dlt = Phase[:,i]-Phase[:,k]
            I_edge[:,e_num] = np.cos(phi_dlt)
            # 复指数形式代码
            # I_edge[:,e_num] = np.exp(1j*(Phase[:,i]-Phase[:,k]))           
            
            e_num = e_num+1
            # 记录每条边对应的两个结点
            node12 = "("+str(i)+","+str(k)+")"
            edges = np.append(edges,node12)
            # 结点1，结点2
            node1,node2 = i,k
            nodes1 = np.append(nodes1,node1)
            nodes2 = np.append(nodes2,node2)
    
    return I_edge,edges,nodes1,nodes2

def Int_delta(ts):
    # T，N，M=时间，通道数/节点数，边数
    Phase = np.angle(signal.hilbert(ts))
    T,N = Phase.shape
    M = int(N*(N-1)/2)

    # 边瞬时强度的时间序列，维度为时间*边数
    I_edge = np.zeros([T,M])
    e_num = 0
    edges = []
    nodes1 = []
    nodes2 = []
    
    for i in range(1,N):
        for j in range(i):
            # 进行希尔伯特变换,得到每个时间点𝑡的瞬时相位信息
            # 计算两点的相位差，得到边的瞬时强度
            # 直接相位做差
            I_edge[:,e_num] = Phase[:,j]-Phase[:,i]
            e_num = e_num+1
            # 记录每条边对应的两个结点
            node12 = "("+str(i)+","+str(j)+")"
            edges = np.append(edges,node12)
            # 结点1，结点2
            node1,node2 = i,j
            nodes1 = np.append(nodes1,node1)
            nodes2 = np.append(nodes2,node2)
    
    return I_edge,edges,nodes1,nodes2

# 直接复现参考文献的相关方法
# 结果并不理想
'''def Int_corr(ts):
    # T，N，M=时间，通道数/节点数，边数
    Phase = np.angle(signal.hilbert(ts))
    T,N = Phase.shape
    M = int(N*(N-1)/2)

    # 边瞬时强度的时间序列，维度为时间*边数
    I_edge = np.zeros([T,M])
    e_num = 0
    edges = []
    
    for i in range(1,N):
        for j in range(i):
            # 进行希尔伯特变换,得到每个时间点𝑡的瞬时相位信息
            # 计算两点的相位差，得到边的瞬时强度
            # 直接相位做差
            I_edge[:,e_num] = zscore(Phase[:,j])*zscore(Phase[:,i])
            e_num = e_num+1
            # 记录每条边对应的两个结点
            nodes = "("+str(i)+","+str(j)+")"
            edges = np.append(edges,nodes)
    
    return I_edge,edges'''

# %%
# 功能：根据边瞬时强度的时间序列I_edge计算eFC矩阵
# 输入：边瞬时强度的时间序列I_edge
# 输出：eFC矩阵
# 使用3种不同的计算形式
# def eFC_mat(I_edge):
#     E_M = np.matrix(I_edge)
#     A = E_M.T*E_M
#     B = np.sqrt(np.diag(A))
#     B = np.matrix(B)
#     C = B.T*B
#     eFC = np.array(A)/np.array(C)
    
#     return eFC

# %%
# 有向边的情况
def eFC_mat_delta(I_edge):
    T,M = I_edge.shape
    eFC = np.zeros([M,M])

    for i in range(M):
        for j in range(M):
            if j<i:
                phi_ddlt = I_edge[:,j]+I_edge[:,i]
            else:
                phi_ddlt = I_edge[:,i]-I_edge[:,j]
            # # 角度差做差
            # eFC[i,j] = np.mean(abs(phi_ddlt))

            # 角度差做差并计算PLV            
            eFC_cos = np.sum(np.cos(phi_ddlt))
            eFC_sin = np.sum(np.sin(phi_ddlt))
            eFC[i,j] = np.sqrt(eFC_cos**2+eFC_sin**2)/T

    return eFC

# %%
# 计算余弦相似度，作为双边的共波动特性
def eFC_mat_cs(I_edge):
    
    eFC = cos_sim(I_edge.T,I_edge.T)

    return eFC

# %%
# 计算皮尔逊相关系数，作为双边的共波动特性
def eFC_mat_pr(I_edge):
    
    eFC = np.corrcoef(I_edge.T,I_edge.T)

    return eFC