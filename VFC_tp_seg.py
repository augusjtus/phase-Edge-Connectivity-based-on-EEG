import numpy as np
import scipy.io as scio
import VFC_dyad

# 功能：将一组dyad（若干个trial）EEG时间序列按时间拼接
def Ts_cct_trl(ts):
    i=0
    # 一组dyad（若干个trial）EEG时间序列的拼接结果
    ts_dyad=[]
    for i in range(ts.shape[0]):
        # ts_trl：当前trial对应的序列
        # ts的格式为被试数*时间点数*通道数
        ts_trl = ts[i]
        # 拼接
        # 结果的维度为通道数*总时间点数
        if i==0:
            ts_dyad=ts_trl
        else:
            ts_dyad=np.r_[ts_dyad,ts_trl]
    # eFC_uptran_bloc_mean = np.mean(eFC_uptran_bloc,axis=1)
    return ts_dyad

# 输入：所有被试的EEG序列的文件路径列表dyad_all，列表中每个元素对应一个被试的EEG序列
# 输出：所有trial的VeFC（eFC矩阵向量化），排列成一个矩阵
def Vefc_dyad_seg(dyad_all,sit,freq,subc,Vtype,uptran_indices,corr_type,tp_seg):
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
    # 群体subc选择
        if subc=='child': # 如果subject的类型为 child
            ts_subc = ts[:,:,0:28]
        elif subc=='adult': # 如果subject的类型为 adult
            ts_subc = ts[:,:,28:56]
    # 序列按时间拼接
        ts_subc_dyad = Ts_cct_trl(ts_subc)
    # EEG时间序列重新分割和拼接    
        # 步长设置为 tp_seg
        tp_seg_list=range(0,ts_subc_dyad.shape[0]+1,tp_seg)
        # ts_dyad_resv是 ts_subc 按照tp_seg的步长分段后保留下来的序列
        tp_resv = tp_seg_list[-1] # 要保留的序列总长度
        ts_dyad_resv = ts_subc_dyad[:tp_resv,:]
        # ts_dyad_rsh是ts_dyad_resv经过reshape过后的序列
        trl_new = int(tp_resv/tp_seg) # 分段过后新的trial数目
        ts_dyad_rsh = ts_dyad_resv.reshape([trl_new,tp_seg,ts_dyad_resv.shape[-1]])
    # 计算eFC的向量形式
        if Vtype=='all': #eFC列向量VeFC
            VeFC_file = VFC_dyad.VeFC_trail_all(ts_dyad_rsh,corr_type)
        elif Vtype=='uptran': #eFC上三角列向量VeFC_uptran
            VeFC_file = VFC_dyad.VeFC_trail_upt(ts_dyad_rsh,uptran_indices,corr_type)
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

# 输入：所有被试的EEG序列的文件路径列表dyad_all，列表中每个元素对应一个被试的EEG序列
# 输出：所有trial的VnFC（nFC矩阵向量化），排列成一个矩阵
def Vnfc_dyad_seg(dyad_all,sit,freq,subc,Vtype,uptran_indices,tp_seg):
    # nFC矩阵向量化的结果记为VnFC
    i=0
    Vnfc_mode=[]
    trn_mode=[] #每种模态trail的数目
    for dyad in dyad_all:
    # 导入mat数据
        data_mat = scio.loadmat(dyad)
        # data_mat的格式为情况*1*频段
        # 选择目标模态对应的数据，ts的格式为被试数*时间点数*通道数
        ts = data_mat[sit][0][freq].T
    # 群体subc选择
        if subc=='child': # 如果subject的类型为 child
            ts_subc = ts[:,:,0:28]
        elif subc=='adult': # 如果subject的类型为 adult
            ts_subc = ts[:,:,28:56]
    # 序列按时间拼接
        ts_subc_dyad = Ts_cct_trl(ts_subc)
    # EEG时间序列重新分割和拼接    
        # 步长设置为 tp_seg
        tp_seg_list=range(0,ts_subc_dyad.shape[0]+1,tp_seg)
        # ts_dyad_resv是 ts_subc 按照tp_seg的步长分段后保留下来的序列
        tp_resv = tp_seg_list[-1] # 要保留的序列总长度
        ts_dyad_resv = ts_subc_dyad[:tp_resv,:]
        # ts_dyad_rsh是ts_dyad_resv经过reshape过后的序列
        trl_new = int(tp_resv/tp_seg) # 分段过后新的trial数目
        ts_dyad_rsh = ts_dyad_resv.reshape([trl_new,tp_seg,ts_dyad_resv.shape[-1]])
    # 计算eFC的向量形式
        if Vtype=='all': #eFC列向量VeFC
            VnFC_file = VFC_dyad.VnFC_trail_all(ts_dyad_rsh)
        elif Vtype=='uptran': #eFC上三角列向量VeFC_uptran
            VnFC_file = VFC_dyad.VnFC_trail_upt(ts_dyad_rsh,uptran_indices)
    # 拼接同一个mode对应的所有文件中的eFC上三角列向量
    # 结果的维度为列向量长度*总的被试数
        trn_mdi=VnFC_file.shape[1]
        if i==0:
            Vnfc_mode=VnFC_file
            trn_mode=trn_mdi
            i=1
        else:
            Vnfc_mode=np.c_[Vnfc_mode,VnFC_file]
            trn_mode=np.c_[trn_mode,trn_mdi]
    return Vnfc_mode,trn_mode