# %% 
# ex2fd后缀表示扩展到文件夹进行遍历
import numpy as np
import scipy.io as scio
import os

from eFC import Int_exp
from eFC import eFC_mat_cs
from eFC import eFC_mat_pr

import VFC_dyad
import Idiff
import VeFC_mst

import classification_accuracy
# %%
# 计算同一组dyad在2种不同拼图场景sit_cmp的VnFC汇总矩阵
def VnFC_motion_dyad(data_mat,motion_comp,freq,subc):
    i=0
    VnFC_motion=[]
    for motion in motion_comp:
    # 选择拼图场景对应的数据，ts的格式为被试数*时间点数*通道数
        ts = data_mat[motion][0][freq].T
    # 群体subc选择
        if subc=='child': # 如果subject的类型为 child
            ts_subc = ts[:,:,0:28]
        elif subc=='adult': # 如果subject的类型为 adult
            ts_subc = ts[:,:,28:56]
    # 计算eFC的向量形式
        VnFC_motioni = VFC_dyad.VnFC_trail_all(ts_subc)
        trn_motioni=VnFC_motioni.shape[1]
    # 拼接VFC矩阵
        if i==0:
            VnFC_motion=VnFC_motioni
            trn_motion=trn_motioni
            i=1
        else:
            VnFC_motion=np.c_[VnFC_motion,VnFC_motioni]
            trn_motion=np.c_[trn_motion,trn_motioni]
                            
    return VnFC_motion,trn_motion

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
    eFC_mean = eFC_all/ts.shape[0]
    return eFC_mean

# %%
# 计算同一组dyad在2种不同拼图场景sit_cmp的VeFC汇总矩阵
def VeFC_mst_motion_dyad(data_mat,sit_cmp,freq,subc,corr_type):
    i=0
    eFC_all_sit=[]

    for siti in sit_cmp:
    # 选择拼图场景对应的数据，ts的格式为被试数*时间点数*通道数
        ts = data_mat[siti][0][freq].T
    # 群体subc选择
        if subc=='child': # 如果subject的类型为 child
            ts_subc = ts[:,:,0:28]
        elif subc=='adult': # 如果subject的类型为 adult
            ts_subc = ts[:,:,28:56]
    # 计算eFC的向量形式
        eFC_mean_all = eFC_dyad_mean(ts_subc,corr_type)
        if len(eFC_all_sit)==0:
            eFC_all_sit=eFC_mean_all
        else:
            eFC_all_sit+=eFC_mean_all
    eFC_all_sit=eFC_all_sit/2
    # 取绝对值
    eFC_mean_abs = np.abs(eFC_all_sit)
    eFC_mean_1d = eFC_mean_abs.reshape([1,len(eFC_mean_abs)**2])
    eFC_mean_1d_inv = -eFC_mean_1d
    eFC_mean_sort = eFC_mean_1d_inv.argsort()
    edge_mst_idx = edge_mst_list(eFC_mean_sort) 
    
    for siti in sit_cmp:
    # 选择拼图场景对应的数据，ts的格式为被试数*时间点数*通道数
        ts = data_mat[siti][0][freq].T
    # 群体subc选择
        if subc=='child': # 如果subject的类型为 child
            ts_subc = ts[:,:,0:28]
        elif subc=='adult': # 如果subject的类型为 adult
            ts_subc = ts[:,:,28:56]        
        VeFC_siti = VeFC_mst.VeFC_mst(ts_subc,corr_type,edge_mst_idx)
        trn_siti=VeFC_siti.shape[1]
    # 拼接VFC矩阵
        if i==0:
            VeFC_sit=VeFC_siti
            trn_sit=trn_siti
            i=1
        else:
            VeFC_sit=np.c_[VeFC_sit,VeFC_siti]
            trn_sit=np.c_[trn_sit,trn_siti]
                            
    return VeFC_sit,trn_sit

def edge_mst_list(eFC_mean_sort,edge_num=378,mst_num=378):
    edge_list1,edge_list2=[],[]

    for mn in range(mst_num):
        sort = eFC_mean_sort[:,edge_num+mn*2][0]
        idx_i = sort//edge_num
        idx_j = sort%edge_num
    # 边列表1，2
        edge_list1 = np.append(edge_list1,idx_i)
        edge_list2 = np.append(edge_list2,idx_j)

    edge_list = np.c_[edge_list1,edge_list2]

    return edge_list

# %%
# 计算同一组dyad在2种不同拼图场景sit_cmp的eFC方法和nFC方法的场景识别性
# Idiff_sit
def Acc_motion_eFC_all(GROUP_DATA_FOLDER,sit_cmp,freq,subc,corr_type,eFC_mean_sort):
    acrcy_list = []
    binary_acrcy_list = []
    for filepath,dirnames,filenames in os.walk(GROUP_DATA_FOLDER):
        for filename in filenames:
            path_mat = os.path.join(filepath,filename)
            # 数据为.mat格式 
            if path_mat[-4:] == '.mat':
                # 导入mat数据
                data_mat = scio.loadmat(path_mat)# data_mat的格式为情况*1*频段
                ## eFC结果
                # edge_mst_idx = edge_mst_list(eFC_mean_sort) 
                VeFC_sitd,trn_sitd = VeFC_mst_motion_dyad(data_mat,sit_cmp,freq,
                                                 subc,corr_type)
                acrcy,_ = classification_accuracy.topk_pearsonr_accuracy(VeFC_sitd,trn_sitd,1)
                binary_acrcy = classification_accuracy.binary_classification_accuracy(VeFC_sitd,trn_sitd)
            # print(acrcy) 
            acrcy_list.append(acrcy)   
            binary_acrcy_list.append(binary_acrcy)
    return acrcy_list,binary_acrcy_list

def Acc_motion_nFC_all(GROUP_DATA_FOLDER,motion_comp,freq,subc,corr_type):
    acrcy_list = []
    binary_acrcy_list = []
    for filepath,dirnames,filenames in os.walk(GROUP_DATA_FOLDER):
        for filename in filenames:
            path_mat = os.path.join(filepath,filename)
            # 数据为.mat格式 
            if path_mat[-4:] == '.mat':
                # 导入mat数据
                data_mat = scio.loadmat(path_mat)# data_mat的格式为情况*1*频段
                ## nFC结果
                VnFC_sitd,trn_sitd = VnFC_motion_dyad(data_mat,motion_comp,freq,
                                                 subc)
                acrcy,_ = classification_accuracy.topk_pearsonr_accuracy(VnFC_sitd,trn_sitd,1)
                binary_acrcy = classification_accuracy.binary_classification_accuracy(VnFC_sitd,trn_sitd)
            # print(acrcy) 
            acrcy_list.append(acrcy)   
            binary_acrcy_list.append(binary_acrcy)
    return acrcy_list,binary_acrcy_list

