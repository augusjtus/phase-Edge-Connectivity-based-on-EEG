# %%
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
import os

# %%
# 功能：遍历group文件夹GROUP_DATA_FOLDER，拼接所有dyad数据文件的路径path_group并输出
# 输入group文件夹GROUP_DATA_FOLDER
# 输出所有dyad数据文件的路径path_group
def file_path_infd(GROUP_DATA_FOLDER):
    path_group=[]
    i=0
# 遍历文件夹
    for filepath,dirnames,filenames in os.walk(GROUP_DATA_FOLDER):
    # 对每个dyad数据文件
        for filename in filenames:
        # 获取路径
            path_mat = os.path.join(filepath,filename)
        # 拼接
            if i==0:
                path_group=path_mat
                i=1
            else:
                path_group=np.append(path_group,path_mat)
    return path_group

# %%
# 功能：计算所有的识别系数I_diff,I_self,I_other，为原始结果
# 输入：所有模态mode拼接好的VeFC_mode,所有模态mode拼接好的trn_mode(每种模态trail的数目)
# 输出：I_diff,I_self,I_other
def get_Idiff(VeFC_mode,trn_mode,type):
    # 计算所有模态mode的总的相关系数矩阵
    if type=='pearsonr':
        R_mode = np.corrcoef(VeFC_mode.T)
        
        i=0
        trn_pre=0
        I_self=[]
        for trn in trn_mode[0]:
            if i==0:
            # 确定前后索引位置
                pre_idx,post_idx=0,trn
            # 提取自相关系数矩阵
                R_mdi=R_mode[pre_idx:post_idx,pre_idx:post_idx]
                lenthi = R_mdi.shape[0]
            # 计算自相关识别系数
                I_selfi = (np.sum(R_mdi)-lenthi)/(lenthi*lenthi-lenthi)
                I_self=I_selfi
                sub_sum = np.sum(R_mdi)
                i=1
            else:
            # 确定前后索引位置
                pre_idx,post_idx=trn_pre,trn_pre+trn
            # 提取自相关系数矩阵
                R_mdi=R_mode[pre_idx:post_idx,pre_idx:post_idx]
                lenthi = R_mdi.shape[0]
            # 计算自相关识别系数
                I_selfi = (np.sum(R_mdi)-lenthi)/(lenthi*lenthi-lenthi)
            # 拼接所有场景的自相关识别系数
                I_self=np.c_[I_self,I_selfi]
                sub_sum = sub_sum+np.sum(R_mdi)
            trn_pre+=trn #记录前一个场景的trail number
            # print(lenthi)
        I_self_mean = np.mean(I_self)
        # 互相关识别系数
        I_other = (np.sum(R_mode)-sub_sum)/(R_mode.shape[0]**2-np.sum(trn_mode[0]**2))

    # 计算协方差矩阵    
    elif type=='cov':
        R_mode = np.cov(VeFC_mode.T)

    # 提取自相关系数矩阵并计算自相关识别系数
        i=0
        I_self=[]
        for trn in trn_mode[0]:
            if i==0:
            # 确定前后索引位置
                pre_idx,post_idx=0,trn
            # 提取自相关系数矩阵
                R_mdi=R_mode[pre_idx:post_idx,pre_idx:post_idx]
                lenthi = R_mdi.shape[0]
            # 计算自相关识别系数
                I_selfi = (np.sum(R_mdi))/(lenthi*lenthi)
                I_self=I_selfi
                sub_sum = np.sum(R_mdi)
                i=1
            else:
            # 确定前后索引位置
                pre_idx,post_idx=trn_pre,trn_pre+trn
            # 提取自相关系数矩阵
                R_mdi=R_mode[pre_idx:post_idx,pre_idx:post_idx]
                lenthi = R_mdi.shape[0]
            # 计算自相关识别系数
                I_selfi = (np.sum(R_mdi))/(lenthi*lenthi)
            # 拼接所有场景的自相关识别系数
                I_self=np.c_[I_self,I_selfi]
                sub_sum = sub_sum+np.sum(R_mdi)
            trn_pre=trn #记录前一个场景的trail number
            # print(lenthi)
        I_self_mean = np.mean(I_self)
        # 互相关识别系数
        I_other = (np.sum(R_mode)-sub_sum)/(R_mode.shape[0]**2-np.sum(trn_mode[0]**2))
    
    I_diff = (I_self_mean-I_other)*100
    # 百分比变化率
    per_change = (I_self_mean-I_other)*100/I_other
    # 百分比差异率
    per_diff = (I_self_mean-I_other)*100*2/(I_self_mean+I_other)
    
    return I_diff,I_self,I_other,I_self_mean,per_change,per_diff

# %%
# 功能：输入包含所有eFC向量的VeFC矩阵，输出其修正矩阵norm_mat
def cal_norm_mat(VeFC_g12):
    n_trl =  VeFC_g12.shape[1]
    norm_vect=np.zeros([n_trl])

    for i in range(VeFC_g12.shape[1]):
        norm_vect[i]=np.linalg.norm(VeFC_g12[:,i])
    norm_vect
    norm_mat=np.zeros([VeFC_g12.shape[1],VeFC_g12.shape[1]])
    for i in range(VeFC_g12.shape[1]):
        for j in range(VeFC_g12.shape[1]):
            if norm_vect[i]<norm_vect[j]:
                norm_mat[i,j]=norm_vect[i]/norm_vect[j]
            else:
                norm_mat[i,j]=norm_vect[j]/norm_vect[i]
    return norm_mat

# 功能：计算所有的识别系数I_diff,I_self,I_other，为经过norm_mat修正后的结果
# 输入：所有模态mode拼接好的VeFC_mode,所有模态mode拼接好的trn_mode(每种模态trail的数目)
# 输出：I_diff,I_self,I_other
def cal_Iden_rvs(VeFC_mode,trn_mode):
    # 计算所有模态mode的总的相关系数矩阵
    R_mode = np.corrcoef(VeFC_mode.T)
    norm_mat = cal_norm_mat(VeFC_mode)
    R_mode = R_mode*norm_mat
    
# 提取自相关系数矩阵并计算自相关识别系数
    i=0
    I_self=[]
    for trn in trn_mode[0]:
        if i==0:
        # 确定前后索引位置
            pre_idx,post_idx=0,trn
        # 提取自相关系数矩阵
            R_mdi=R_mode[pre_idx:post_idx,pre_idx:post_idx]
            lenthi = R_mdi.shape[0]
        # 计算自相关识别系数
            I_selfi = (np.sum(R_mdi)-lenthi)/(lenthi*lenthi-lenthi)
            I_self=I_selfi
            sub_sum = np.sum(R_mdi)
            i=1
        else:
        # 确定前后索引位置
            pre_idx,post_idx=trn_pre,trn_pre+trn
        # 提取自相关系数矩阵
            R_mdi=R_mode[pre_idx:post_idx,pre_idx:post_idx]
            lenthi = R_mdi.shape[0]
        # 计算自相关识别系数
            I_selfi = (np.sum(R_mdi)-lenthi)/(lenthi*lenthi-lenthi)
        # 拼接所有场景的自相关识别系数
            I_self=np.c_[I_self,I_selfi]
            sub_sum = sub_sum+np.sum(R_mdi)
        trn_pre=trn #记录前一个场景的trail number
        # print(lenthi)
    I_self_mean = np.mean(I_self)
    # 互相关识别系数
    I_other = (np.sum(R_mode)-sub_sum)/(R_mode.shape[0]**2-np.sum(trn_mode[0]**2))
    I_diff = (I_self_mean-I_other)*100
    
    return I_diff,I_self,I_other


