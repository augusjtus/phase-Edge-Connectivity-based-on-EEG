import numpy as np
import Idiff

# 功能：将样本dyad_smp中所有dyad对应的VFC序列提取出来，组成一个比较小的VFC矩阵
def VeFc_resmp(dyad_smp,trn_mode,VeFC_mode):
    i=0
    VFC_smp=[] # 样本dyad_smp对应的VFC矩阵
    trn_smp=[] # 样本dyad_smp对应trail数目序列

    for dyad_i in dyad_smp:
    # 二人组i对应的trial数目
        trn_i = trn_mode[:,dyad_i]
    # 二人组i的trial在VFC矩阵中对应的位置
    # 前后索引
        pre_idx_i,post_idx_i=int(np.sum(trn_mode[:,:dyad_i])),int(np.sum(trn_mode[:,:dyad_i])+trn_i)
        VFC_mode_i = VeFC_mode[:,pre_idx_i:post_idx_i]
    # 拼接dyad_smp中所有二人组i的trial数目、VFC序列
        if i==0:
            VFC_smp=VFC_mode_i
            trn_smp=trn_i
            i=1
        else:
            VFC_smp=np.c_[VFC_smp,VFC_mode_i]
            trn_smp=np.c_[trn_smp,trn_i]
    
    return VFC_smp,trn_smp

# 功能：将样本dyad_smp中所有dyad对应的VFC序列提取出来，组成一个比较小的VFC矩阵
def VnFc_resmp(dyad_smp,trn_mode,VnFC_mode):
    i=0
    VFC_smp=[] # 样本dyad_smp对应的VFC矩阵
    trn_smp=[] # 样本dyad_smp对应trail数目序列

    for dyad_i in dyad_smp:
    # 二人组i对应的trial数目
        trn_i = trn_mode[:,dyad_i]
    # 二人组i的trial在VFC矩阵中对应的位置
    # 前后索引
        pre_idx_i,post_idx_i=int(np.sum(trn_mode[:,:dyad_i])),int(np.sum(trn_mode[:,:dyad_i])+trn_i)
        VFC_mode_i = VnFC_mode[:,pre_idx_i:post_idx_i]
    # 拼接dyad_smp中所有二人组i的trial数目、VFC序列
        if i==0:
            VFC_smp=VFC_mode_i
            trn_smp=trn_i
            i=1
        else:
            VFC_smp=np.c_[VFC_smp,VFC_mode_i]
            trn_smp=np.c_[trn_smp,trn_i]
    
    return VFC_smp,trn_smp

# 功能：对VFC汇总矩阵进行bootstrap抽样，并计算每次抽样的识别指标Idiff
# 输入：res_time抽样次数,yita抽样率，VeFC_mode,VnFC_mode：汇总矩阵，trn_mode汇总trial数目
def Idiff_boot(res_time,yita,VeFC_mode,VnFC_mode,trn_mode):
    # dyad二人组的总数目
    dyad_num = trn_mode.shape[1]
    # dyad二人组编号索引
    dyad_idx = range(dyad_num)
    # 重抽样（默认：30% = 15组）
    resmp_num = int(dyad_num*yita)
    
    k=0
    for _ in range(res_time):  
    # 不重复采样
        dyad_smp = np.random.choice(dyad_idx, size=resmp_num, replace=False)
    
    ## eFC结果
    # 将样本dyad_smp中所有dyad对应的VeFC序列提取出来
        VeFC_smp,trn_smp = VeFc_resmp(dyad_smp,trn_mode,VeFC_mode)
    # 计算样本dyad_smp的识别性指标I_diff
        I_diff,I_self,I_other,I_self_mean,per_change,per_diff=Idiff.get_Idiff(VeFC_smp,
                                                                         trn_smp,'pearsonr')
    # 保留两位小数并记录数据
        I_concat = [I_diff,I_self_mean,I_other,per_change,per_diff]
        I_eFC1 = np.around(I_concat,2)

    ## nFC结果
    # 将样本dyad_smp中所有dyad对应的VnFC序列提取出来
        VnFC_smp,trn_smp = VnFc_resmp(dyad_smp,trn_mode,VnFC_mode)
    # 计算样本dyad_smp的识别性指标I_diff
        I_diff,I_self,I_other,I_self_mean,per_change,per_diff=Idiff.get_Idiff(VnFC_smp,
                                                                         trn_smp,'pearsonr')
    # 保留两位小数并记录数据
        I_concat = [I_diff,I_self_mean,I_other,per_change,per_diff]
        I_nFC1 = np.around(I_concat,2)
    
    # 将所有二人组对的指标拼接
        if k==0:
            I_eFC_all=I_eFC1
            I_nFC_all=I_nFC1
            k=1
        else:
            I_eFC_all=np.c_[I_eFC_all,I_eFC1]
            I_nFC_all=np.c_[I_nFC_all,I_nFC1]
    return I_eFC_all,I_nFC_all