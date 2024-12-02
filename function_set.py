import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics

## Kmeans score
def Kmeans_score(dyad,VeFC,trn):
    # 做kmeans得到预测标签值
    kmeans = KMeans(n_clusters=dyad.shape[0], random_state=0, n_init=10).fit(VeFC.T)

# 拼接得到真实标签值
    true_labels = get_true_labels(trn) 

# 计算预测标签和真实标签吻合程度的评价指标
    rand_score = metrics.adjusted_rand_score(true_labels, kmeans.labels_) #兰德系数 
    mi_score = metrics.adjusted_mutual_info_score(true_labels, kmeans.labels_) #互信息指数
    fm_score = metrics.fowlkes_mallows_score(true_labels, kmeans.labels_) #FM分数
    print('%.2f'%rand_score,'%.2f'%mi_score,'%.2f'%fm_score)

    # print(true_labels,kmeans.labels_)

# 根据trn获得真实标签
def get_true_labels(trn):
    
    true_labels=[]
    for i in range(trn.shape[1]):
        true_label=[i]*trn[0][i]
        if i==0:
            true_labels=true_label
        else:
            true_labels=np.r_[true_labels,true_label]
    true_labels = np.array(true_labels) 

    return true_labels

def top1_sim_accuracy(VFC,trn):
    # 计算最大值对应的标签
    pearsonr_max_list = []
    for trl_idx0 in range(VFC.shape[1]):
        pearsonr_list = []
        for trl_idx1 in range(VFC.shape[1]):
            if trl_idx1 != trl_idx0:
                pearsonr = np.corrcoef(VFC[:,trl_idx0],VFC[:,trl_idx1])[0,1]
                pearsonr_list.append(pearsonr)
        pearsonr_max_list.append(pearsonr_list.index(max(pearsonr_list)))

# 计算真实标签
    true_labels=[]
    for i in range(trn.shape[1]):
        true_label=[i]*trn[0][i]
        if i==0:
            true_labels=true_label
        else:
            true_labels=np.r_[true_labels,true_label]
    true_labels = np.array(true_labels) 

# 计算正确的比例
    labels_comp = true_labels[pearsonr_max_list]-true_labels
    acrcy = np.count_nonzero(labels_comp==0)/len(labels_comp)

    return acrcy

# get the index of topK largest value in list
def get_topk_idx(list,k):
    list_sorted = sorted(list,reverse=True)
    topk_idx=[]
    for elem in list_sorted[:k]:
        topk_idx.append(list.index(elem))
    topk_idx_ary = np.array(topk_idx)
    return topk_idx_ary

def topk_sim_accuracy(VFC,trn,k):
    # 计算最大值对应的标签
    pearsonr_topk_idx_list = []
    for trl_idx0 in range(VFC.shape[1]):
        pearsonr_list = []
        for trl_idx1 in range(VFC.shape[1]):
            if trl_idx1 != trl_idx0:
                pearsonr = np.corrcoef(VFC[:,trl_idx0],VFC[:,trl_idx1])[0,1]
                pearsonr_list.append(pearsonr)
        pearsonr_topk_idx = get_topk_idx(pearsonr_list,k)
        if trl_idx0==0:
            pearsonr_topk_idx_list=pearsonr_topk_idx
        else:
            pearsonr_topk_idx_list=np.c_[pearsonr_topk_idx_list,pearsonr_topk_idx]


# 计算真实标签
    true_labels = get_true_labels(trn)
    # 比较标签，数值为0表示相等
    labels_comp = np.ones(true_labels.shape)
    for i in range(k):
        labels_comp_i = true_labels[list(pearsonr_topk_idx_list[i,:])]-true_labels
        labels_comp *= labels_comp_i

# 计算正确的比例
    acrcy = np.count_nonzero(labels_comp==0)/len(labels_comp)

    return acrcy,pearsonr_topk_idx_list

# pearsonr_top3idx_list,true_labels = function_set.top3_sim_accuracy(VnFC_mode,trn_mode)
# labels_comp0 = true_labels[list(pearsonr_top3idx_list[0,:])]-true_labels
# labels_comp1 = true_labels[list(pearsonr_top3idx_list[1,:])]-true_labels
# labels_comp2 = true_labels[list(pearsonr_top3idx_list[2,:])]-true_labels
# labels_comp = labels_comp0*labels_comp1*labels_comp2

# acrcy = np.count_nonzero(labels_comp==0)/len(labels_comp)
# acrcy