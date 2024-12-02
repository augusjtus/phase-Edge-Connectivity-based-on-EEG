# %% 
import numpy as np
import scipy.signal as signal
import os
import scipy.io as scio

from eFC import Int_exp
from eFC import eFC_mat_cs
from eFC import eFC_mat_pr
import nFC
import Idiff
import MST_Graph

# %%
# 获得矩阵上三角列向量的位置
# 输入：结点数
# 输出：矩阵上三角列向量的位置数组
def eFC_dyad_mean(ts,corr_type):
    eFC_all=[]
    for i in range(ts.shape[0]):
        E = Int_exp(ts[i])[0]
        # 选择相关系数的类型
        if corr_type=='cos':
            eFC = eFC_mat_cs(E)
        elif corr_type=='pearson':
            eFC = eFC_mat_pr(E)
        # 拼接同一个mat文件中所有的eFC矩阵
        # 结果的维度为列向量长度*被试数
        if i==0:
            eFC_all=eFC
        else:
            eFC_all=eFC_all+eFC
    # eFC_mean = eFC_all/ts.shape[0]
    return eFC_all

def eFC_all_mean(dyad_all,sit,freq,subc,corr_type,tp):
    # eFC矩阵向量化的结果记为VeFC
    i=0
    eFC_alld=[]

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
        eFC_dyad = eFC_dyad_mean(ts_subc,corr_type)
    # 拼接同一个mode对应的所有文件中的eFC上三角列向量
    # 结果的维度为列向量长度*总的被试数
        if i==0:
            eFC_alld=eFC_dyad
            i=1
        else:
            eFC_alld=eFC_alld+eFC_dyad
    eFC_mean_all = eFC_alld/dyad_all.shape[0] 

    return eFC_mean_all

def get_edge_mst_idx(dyad_all,sit,freq,subc,corr_type,tp,edge_num):    
    eFC_mean_all = eFC_all_mean(dyad_all,sit,freq,subc,corr_type,tp)
# 取绝对值
    eFC_mean_abs = np.abs(eFC_mean_all)
# 取1-原值：原值越大则1-原值越小，用于之后的最小生成树MST
    eFC_abs_inv = 1-eFC_mean_abs
    edge_list1,edge_list2=[],[]
    for edge1 in range(1,edge_num):
        for edge2 in range(edge1):
        # 边列表1，2
            edge_list1 = np.append(edge_list1,edge1)
            edge_list2 = np.append(edge_list2,edge2)
    eFC_Graph = MST_Graph.Graph()

    for i in range(edge_list1.shape[0]):
        e1,e2=edge_list1[i],edge_list2[i]
        e1,e2=int(e1),int(e2)
        eFC_Graph.add_edge(e1,e2,-eFC_mean_abs[e1,e2])
    
    edge_mst_list = eFC_Graph.prim()
    edge_mst_ary = np.array(edge_mst_list)[:]
    edge_mst_idx = edge_mst_ary[:,:2]

    return edge_mst_idx

# 函数功能：将矩阵中MST路径对应坐标的元素提取出来，作为向量
# 输入：eFC_mean_abs 目标矩阵,edge_mst_idx MST路径对应坐标
# 输出：MST向量
def mst_array(eFC_mean_abs,edge_mst_idx):
    k=0
    for (i,j) in edge_mst_idx[:,:]:
    # 坐标变为整数
        idxi,idxj = int(i),int(j)
    # 拼接对应坐标的元素
        if k==0:
            VeFC_mst=eFC_mean_abs[idxi,idxj]
            k=1
        else:
            VeFC_mst=np.append(VeFC_mst,eFC_mean_abs[idxi,idxj])
# 将1维向量reshape为2维数组
    VeFC_mst.reshape((VeFC_mst.shape[0],1))
    return VeFC_mst

# 函数功能：将所有eFC矩阵的MST向量汇总成矩阵
# 输入：ts EEG时间序列,corr_type 相关分析类型,
# edge_mst_idx MST路径对应坐标
def VeFC_mst(ts,corr_type,edge_mst_idx):
    VeFC_mst_dyad=[]
    for i in range(ts.shape[0]):
        E = Int_exp(ts[i])[0]
        # 选择相关系数的类型
        if corr_type=='cos':
            eFC = eFC_mat_cs(E)
        elif corr_type=='pearson':
            eFC = eFC_mat_pr(E)
        # 拼接同一个mat文件中所有的eFC矩阵
        # 结果的维度为列向量长度*被试数
        VeFC_msti = mst_array(eFC,edge_mst_idx)
        if i==0:
            VeFC_mst_dyad=VeFC_msti
        else:
            VeFC_mst_dyad=np.c_[VeFC_mst_dyad,VeFC_msti]

    return VeFC_mst_dyad

def VeFC_mst_all(dyad_all,sit,freq,subc,corr_type,tp,edge_mst_idx):
    # eFC矩阵向量化的结果记为VeFC
    i=0
    VeFC_mst_alld=[]

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
        VeFC_mst_dyad = VeFC_mst(ts_subc,corr_type,edge_mst_idx)
    # 拼接同一个mode对应的所有文件中的eFC上三角列向量
    # 结果的维度为列向量长度*总的被试数
        if i==0:
            VeFC_mst_alld=VeFC_mst_dyad
            i=1
        else:
            VeFC_mst_alld=np.c_[VeFC_mst_alld,VeFC_mst_dyad]

    return VeFC_mst_alld
