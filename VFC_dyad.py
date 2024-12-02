# %% 
# ex2fd后缀表示扩展到文件夹进行遍历
import numpy as np
import scipy.io as scio

from eFC import Int_exp
from eFC import eFC_mat_cs
from eFC import eFC_mat_pr
from nFC import nFC_exp

# %%
# 获得矩阵上三角列向量的位置
# 输入：结点数
# 输出：矩阵上三角列向量的位置数组
def uptran_mask(node_num):
    A = np.ones([node_num,node_num])
    E = np.eye(node_num)
    U = np.triu(A)-E
    vector_mask = U.flatten("F")
    uptran_indices = np.where(vector_mask > 0)[0]
    
    return uptran_indices

# %%
# 功能：拼接同一个dyad所有的trail对应的eFC列向量VeFC
# 输入：ts,itype=时间序列，边功能连接强度计算方式（PLV、脑电信号、相位差）
# 结果的维度为列向量长度*被试数
def VeFC_trail_all(ts,corr_type):
    VeFC_bloc=[]
    for i in range(ts.shape[0]):
        E = Int_exp(ts[i])[0]
        # 选择相关系数的类型
        if corr_type=='cos':
            eFC = eFC_mat_cs(E)
        elif corr_type=='pearson':
            eFC = eFC_mat_pr(E)
        # 获得eFC列向量eFC_flatten：
        VeFC = eFC.flatten("F")
        # 拼接同一个mat文件中所有的eFC列向量
        # 结果的维度为列向量长度*被试数
        if i==0:
            VeFC_bloc=VeFC
        else:
            VeFC_bloc=np.c_[VeFC_bloc,VeFC]
    # eFC_uptran_bloc_mean = np.mean(eFC_uptran_bloc,axis=1)
    return VeFC_bloc

# 功能：拼接同一个dyad所有的trail对应的eFC上三角列向量VeFC_uptran
# 输入：ts,itype=时间序列，边功能连接强度计算方式（PLV、脑电信号、相位差）
# corr_type：相关系数的类型
# 结果的维度为列向量长度*被试数
def VeFC_trail_upt(ts,uptran_indices,corr_type):
    VeFC_uptran_bloc=[]
    for i in range(ts.shape[0]):
        E = Int_exp(ts[i])[0]
        # 选择相关系数的类型
        if corr_type=='cos':
            eFC = eFC_mat_cs(E)
        elif corr_type=='pearson':
            eFC = eFC_mat_pr(E)
        # 获得矩阵的上三角列向量：
        # 提取矩阵的上三角元素（不包含主对角线），按照列的顺序将其整合成向量
        eFC_flatten = eFC.flatten("F")
        VeFC_uptran = eFC_flatten[uptran_indices]
        # 拼接同一个mat文件中所有的eFC上三角列向量
        # 结果的维度为列向量长度*被试数
        if i==0:
            VeFC_uptran_bloc=VeFC_uptran
        else:
            VeFC_uptran_bloc=np.c_[VeFC_uptran_bloc,VeFC_uptran]
    # eFC_uptran_bloc_mean = np.mean(eFC_uptran_bloc,axis=1)
    return VeFC_uptran_bloc

# %%
# 功能：拼接同一个mat文件中所有的trail对应的 nFC 列向量 VnFC
# itype：计算FC矩阵的方法
# 结果的维度为列向量长度*被试数
def VnFC_trail_all(ts):
    VnFC_bloc=[]
    for i in range(ts.shape[0]):
        nFC = nFC_exp(ts[i])
        # 获得 nFC 列向量 VnFC ：
        VnFC = nFC.flatten("F")
        # 拼接同一个mat文件中所有的 nFC 列向量
        # 结果的维度为列向量长度*被试数
        if i==0:
            VnFC_bloc=VnFC
        else:
            VnFC_bloc=np.c_[VnFC_bloc,VnFC]
    # eFC_uptran_bloc_mean = np.mean(eFC_uptran_bloc,axis=1)
    return VnFC_bloc

def VnFC_trail_upt(ts,uptran_indices):
    VnFC_bloc=[]
    for i in range(ts.shape[0]):
        nFC = nFC_exp(ts[i])
        # 获得 nFC 列向量 VnFC ：
        nFC_flatten = nFC.flatten("F")
        VnFC = nFC_flatten[uptran_indices]
        # 拼接同一个mat文件中所有的 nFC 列向量
        # 结果的维度为列向量长度*被试数
        if i==0:
            VnFC_bloc=VnFC
        else:
            VnFC_bloc=np.c_[VnFC_bloc,VnFC]
    # eFC_uptran_bloc_mean = np.mean(eFC_uptran_bloc,axis=1)
    return VnFC_bloc

# %%
# 功能：拼接所有dyad对应的所有trail的VeFC向量，得到VeFC向量汇总矩阵
def VeFC_dyad_all(dyad_all,sit,freq,subc,Vtype,uptran_indices,corr_type,tp):
    # eFC矩阵向量化的结果记为VeFC
    i=0
    VeFC_mode=[]
    trn_mode=[] #每种模态trail的数目
    for dyad in dyad_all:
    # 导入mat数据
        data_mat = scio.loadmat(dyad)
    # data_mat的格式为情况*1*频段
    # 选择目标模态对应的数据，ts的格式为被试数*时间点数*通道数
        ts = data_mat[sit][0][freq].T
    # 时间点选择
        ts = ts[:,0:tp,:]
    # 群体subc选择
        if subc=='child': # 如果subject的类型为 child
            ts_subc = ts[:,:,0:28]
        elif subc=='adult': # 如果subject的类型为 adult
            ts_subc = ts[:,:,28:56]
    # 计算eFC的向量形式
        if Vtype=='all': #eFC列向量VeFC
            VeFC_file = VeFC_trail_all(ts_subc,corr_type)
        elif Vtype=='uptran': #eFC上三角列向量VeFC_uptran
            VeFC_file = VeFC_trail_upt(ts_subc,uptran_indices,corr_type)
    # 拼接同一个mode对应的所有文件中的eFC上三角列向量
    # 结果的维度为列向量长度*总的被试数
        trn_mdi=VeFC_file.shape[1]
        if i==0:
            VeFC_mode=VeFC_file
            trn_mode=trn_mdi
            i=1
        else:
            VeFC_mode=np.c_[VeFC_mode,VeFC_file]
            trn_mode=np.c_[trn_mode,trn_mdi]
    return VeFC_mode,trn_mode

# %%
# 功能：拼接所有dyad对应的所有trail的VnFC向量，得到VnFC向量汇总矩阵
def VnFC_dyad_all(dyad_all,sit,freq,subc,Vtype,uptran_indices,tp):
    # eFC矩阵向量化的结果记为VeFC
    i=0
    VnFC_mode=[]
    trn_mode=[] #每种模态trail的数目
    for dyad in dyad_all:
    # 导入mat数据
        data_mat = scio.loadmat(dyad)
        # data_mat的格式为情况*1*频段
        # 选择目标模态对应的数据，ts的格式为被试数*时间点数*通道数
        ts = data_mat[sit][0][freq].T
        # 时间点选择
        ts = ts[:,0:tp,:]
        # 群体subc选择
        if subc=='child': # 如果subject的类型为 child
            ts_subc = ts[:,:,0:28]
        elif subc=='adult': # 如果subject的类型为 adult
            ts_subc = ts[:,:,28:56]
        # 计算eFC的向量形式
        if Vtype=='all': #eFC列向量VeFC
            VnFC_file = VnFC_trail_all(ts_subc)
        elif Vtype=='uptran': #eFC上三角列向量VeFC_uptran
            VnFC_file = VnFC_trail_upt(ts_subc,uptran_indices)
        # 拼接同一个mode对应的所有文件中的eFC上三角列向量
        # 结果的维度为列向量长度*总的被试数
        trn_mdi=VnFC_file.shape[1]
        if i==0:
            VnFC_mode=VnFC_file
            trn_mode=trn_mdi
            i=1
        else:
            VnFC_mode=np.c_[VnFC_mode,VnFC_file]
            trn_mode=np.c_[trn_mode,trn_mdi]

    return VnFC_mode,trn_mode

