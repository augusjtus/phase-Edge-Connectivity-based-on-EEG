# %%
import numpy as np
import os
import scipy.io as scio

import MST_Graph
import VeFC_mst

import Idiff

## VeFC_sit_dyad函数改为eFC_all_mean的形式
# 计算同一组dyad在各拼图场景sit_cmp中所有trial的eFC平均值
def eFC_sit_mean(data_mat,sit_cmp,freq,subc,corr_type):
    i=0
    eFC_alld=[]
    trn_num=0

    for siti in sit_cmp:
    # 选择拼图场景对应的数据，ts的格式为被试数*时间点数*通道数
        ts = data_mat[siti][0][freq].T
    # 群体subc选择
        if subc=='child': # 如果subject的类型为 child
            ts_subc = ts[:,:,0:28]
        elif subc=='adult': # 如果subject的类型为 adult
            ts_subc = ts[:,:,28:56]
        # 计算eFC的向量形式
        eFC_dyad = VeFC_mst.eFC_dyad_mean(ts_subc,corr_type)
        trn_num += ts.shape[0]
    # 拼接同一个mode对应的所有文件中的eFC上三角列向量
    # 结果的维度为列向量长度*总的被试数
        if i==0:
            eFC_alld=eFC_dyad
            i=1
        else:
            eFC_alld=eFC_alld+eFC_dyad
    
    eFC_mean_all = eFC_alld/trn_num 
                            
    return eFC_mean_all

def VeFC_sit_mst(data_mat,sit_cmp,freq,subc,corr_type,edge_mst_idx):
    # eFC矩阵向量化的结果记为VeFC
    i=0
    VeFC_mst_alld=[]

    for siti in sit_cmp:
    # 选择拼图场景对应的数据，ts的格式为被试数*时间点数*通道数
        ts = data_mat[siti][0][freq].T
    # 群体subc选择
        if subc=='child': # 如果subject的类型为 child
            ts_subc = ts[:,:,0:28]
        elif subc=='adult': # 如果subject的类型为 adult
            ts_subc = ts[:,:,28:56]
    # 计算eFC的向量形式
        VeFC_mst_dyad = VeFC_mst.VeFC_mst(ts_subc,corr_type,edge_mst_idx)
        trn_siti=VeFC_mst_dyad.shape[1]
    # 拼接同一个mode对应的所有文件中的eFC上三角列向量
    # 结果的维度为列向量长度*总的被试数
        if i==0:
            VeFC_mst_alld=VeFC_mst_dyad
            trn_sit=trn_siti
            i=1
        else:
            VeFC_mst_alld=np.c_[VeFC_mst_alld,VeFC_mst_dyad]
            trn_sit=np.c_[trn_sit,trn_siti]

    return VeFC_mst_alld,trn_sit

# 计算同一组dyad的VeFC汇总矩阵经MST降维后的场景识别系数Idiff_sit_MST
def Idiff_sit_MST(data_mat,sit_cmp,freq,subc,corr_type,edge_num):
    # #### 1、计算出所有eFC矩阵的均值矩阵eFC_mean_all
    eFC_mean_all = eFC_sit_mean(data_mat,sit_cmp,freq,subc,corr_type)
    # 取绝对值
    eFC_mean_abs = np.abs(eFC_mean_all)

    # #### 2、根据均值矩阵eFC_mean_all计算最小生成树MST对应的矩阵编号edge_mst_idx
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
        # 取-原值：原值越大则-原值越小，用于之后的最小生成树MST
        eFC_Graph.add_edge(e1,e2,-eFC_mean_abs[e1,e2])
    edge_mst_list = eFC_Graph.prim()
    edge_mst_ary = np.array(edge_mst_list)[:]
    edge_mst_idx = edge_mst_ary[:,:2]

    # #### 3、计算MST降维后的eFC向量汇总矩阵VeFC_mst_alld
    VeFC_mst_alld,trn_sit = VeFC_sit_mst(data_mat,sit_cmp,freq,subc,corr_type,
                                edge_mst_idx)
    I_diff,I_self,I_other,I_self_mean,per_change,per_diff=Idiff.get_Idiff(VeFC_mst_alld,trn_sit,'pearsonr')
    # 保留两位小数并记录数据
    I_concat = [I_diff,I_self_mean,I_other,per_change,per_diff]
    I_eFC_mst = np.around(I_concat,2)

    return I_eFC_mst

def Idiff_sit_all_MST(GROUP_DATA_FOLDER,sit_cmp,freq,subc,corr_type,edge_num):
    k=0
    for filepath,dirnames,filenames in os.walk(GROUP_DATA_FOLDER):
        for filename in filenames:
            path_mat = os.path.join(filepath,filename)
            # 数据为.mat格式 
            if path_mat[-4:] == '.mat':
                # 导入mat数据
                data_mat = scio.loadmat(path_mat)# data_mat的格式为情况*1*频段
                I_eFC_dyad = Idiff_sit_MST(data_mat,sit_cmp,freq,subc,
                                           corr_type,edge_num)
            # 拼接各个dyad的Idiff
            if k==0:
                I_eFC_all=I_eFC_dyad
                k=1
            else:
                I_eFC_all=np.c_[I_eFC_all,I_eFC_dyad]
    
    return I_eFC_all
