import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

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

def KMeans_s_ch_score(VnFC_subj,trn_subj):
    VnFC_subj_T = VnFC_subj.T

# 进行K-means聚类
    kmeans = KMeans(n_clusters=trn_subj.shape[1], random_state=42)
    labels = kmeans.fit_predict(VnFC_subj_T)

# 计算类内离差
    intra_cluster_distance = 0
    for i in range(trn_subj.shape[1]):
        cluster_points = VnFC_subj_T[labels == i]
        centroid = kmeans.cluster_centers_[i]
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        intra_cluster_distance += np.sum(distances)
    intra_cluster_distance /= VnFC_subj_T.shape[0]
    print(f"类内离差: {intra_cluster_distance:.2f}")

# 计算类间离差
    inter_cluster_distance = 0
    centroids = kmeans.cluster_centers_
    for i in range(trn_subj.shape[1]):
        for j in range(i+1, trn_subj.shape[1]):
            distance = np.linalg.norm(centroids[i] - centroids[j])
            inter_cluster_distance += distance
    inter_cluster_distance /= trn_subj.shape[1]
    print(f"类间离差: {inter_cluster_distance:.2f}")

# 使用其他指标评估聚类效果
    silhouette = silhouette_score(VnFC_subj_T, labels)
    calinski_harabasz = calinski_harabasz_score(VnFC_subj_T, labels)
    print(f"轮廓系数: {silhouette:.2f}")
    print(f"Calinski-Harabasz指数: {calinski_harabasz:.2f}")

## Agg_cluster score
# 创建层次聚类对象，设置聚类的类别数量
def Agg_cluster_score(VnFC_subj,trn_subj):
    n_clusters = trn_subj.shape[1]
    clustering = AgglomerativeClustering(n_clusters=n_clusters)

# 执行层次聚类
    clusters = clustering.fit_predict(VnFC_subj)

# 假设前面已经有了data（数据向量）和clusters（聚类结果）
    silhouette_avg = silhouette_score(VnFC_subj, clusters)
    print("Silhouette Coefficient:", silhouette_avg)

# 绘制聚类结果
    plt.scatter(VnFC_subj[:, 0], VnFC_subj[:, 1], c=clusters)
    plt.title("Hierarchical Clustering Results")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


## topK accuracy
def get_true_labels(trn):
    """    
    get_true_labels
    """

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
    topk_value=[]
    for elem in list_sorted[:k]:
        topk_idx.append(list.index(elem))
        topk_value.append(elem)
    topk_idx_ary = np.array(topk_idx)
    topk_value_ary = np.array(topk_value)
    return topk_idx_ary,topk_value_ary

def topk_sim_accuracy(VFC,trn,k):
    # 计算最大值对应的标签
    pearsonr_topk_idx_list = []
    for trl_idx0 in range(VFC.shape[1]):
        pearsonr_list = []
        for trl_idx1 in range(VFC.shape[1]):
            if trl_idx1 != trl_idx0:
                pearsonr = np.corrcoef(VFC[:,trl_idx0],VFC[:,trl_idx1])[0,1]
                pearsonr_list.append(pearsonr)
        pearsonr_topk_idx,pearsonr_topk_value = get_topk_idx(pearsonr_list,k)
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


## topK accuracy with subject_mean VFC and subject_mean Idiff
# get the mean VFC of the same subject 
def get_VFC_subj_mean(VFC_subj,trn_subj):
    trn_pre = 0 
    VFC_subj_mean = []
    for trn in trn_subj[0]:
    # 确定前后索引位置
        pre_idx,post_idx=trn_pre,trn_pre+trn
    # 提取自相关系数矩阵
        VFC_subji=VFC_subj[:,pre_idx:post_idx]
        VFC_subji_mean = np.mean(VFC_subji,axis=1)
        if trn_pre==0:
            VFC_subj_mean = VFC_subji_mean
        else:
            VFC_subj_mean = np.c_[VFC_subj_mean,VFC_subji_mean]
        trn_pre+=trn #记录前一个场景的trail number
    # print(VFC_subj_mean.shape)
    return VFC_subj_mean

# topK accuracy with subject_mean VFC
def topk_sim_accuracy_meanVFC(VFC,trn,k):
    
    VFC_subj_mean = get_VFC_subj_mean(VFC,trn)
    # 计算真实标签
    true_labels = get_true_labels(trn)
    true_mean_labels = np.array(range(VFC_subj_mean.shape[1]))

    # 计算最大值对应的标签
    pearsonr_topk_idx_list = []
    for trl_idx0 in range(VFC.shape[1]):
        pearsonr_list = []
        for trl_idx1 in range(VFC_subj_mean.shape[1]):
            VFC_idx0 = VFC[:,trl_idx0]
            VFC_subj_mean_idx1 = VFC_subj_mean[:,trl_idx1]
            if true_labels[trl_idx0] == trl_idx1:
                VFC_subj_mean_idx1 = (VFC_subj_mean_idx1*trn[0][trl_idx1]-VFC_idx0)/(trn[0][trl_idx1]-1)
            pearsonr = np.corrcoef(VFC_idx0,VFC_subj_mean_idx1)[0,1]
            pearsonr_list.append(pearsonr)
        pearsonr_topk_idx,pearsonr_topk_value = get_topk_idx(pearsonr_list,k)
        if trl_idx0==0:
            pearsonr_topk_idx_list=pearsonr_topk_idx
        else:
            pearsonr_topk_idx_list=np.c_[pearsonr_topk_idx_list,pearsonr_topk_idx]


    # 比较标签，数值为0表示相等
    labels_comp = np.ones(true_labels.shape)
    for i in range(k):
        labels_comp_i = true_mean_labels[list(pearsonr_topk_idx_list[i,:])]-true_labels
        labels_comp *= labels_comp_i

# 计算正确的比例
    acrcy = np.count_nonzero(labels_comp==0)/len(labels_comp)

    return acrcy,pearsonr_topk_idx_list

def get_pearsonr_subj_mean(pearsonr_list,trn_subj,trl_idx0):
    trn_pre = 0 
    pearsonr_subj_mean = []
    for trn in trn_subj[0]:
    # 确定前后索引位置
        pre_idx,post_idx=trn_pre,trn_pre+trn
    # 提取自相关系数矩阵
        pearsonr_subji=pearsonr_list[pre_idx:post_idx]
        # 如果trl_idx0在pre_idx:post_idx之间，即排序为trl_idx0的vfc和自己的相关，数值为1
        if pre_idx<= trl_idx0 & trl_idx0<=post_idx:
            pearsonr_subji_mean = (np.sum(pearsonr_subji)-1)/(len(pearsonr_subji)-1)
        else:
            pearsonr_subji_mean = np.mean(pearsonr_subji)
        pearsonr_subj_mean.append(pearsonr_subji_mean)
        trn_pre+=trn #记录前一个场景的trail number

    return pearsonr_subj_mean

def get_pearsonr_allothers_mean(pearsonr_list,trn_subj,trl_idx0):
    trn_pre = 0 
    # 个体内的平均相关系数 pearsonr_subji_mean
    for trn in trn_subj[0]:
    # 确定前后索引位置
        pre_idx,post_idx=trn_pre,trn_pre+trn
        # 如果trl_idx0在pre_idx:post_idx之间，即排序为trl_idx0的vfc对应的subject
        if pre_idx<= trl_idx0 & trl_idx0<=post_idx:
            pearsonr_subji=pearsonr_list[pre_idx:post_idx]
            pearsonr_subji_mean = (np.sum(pearsonr_subji)-1)/(len(pearsonr_subji)-1)
        trn_pre+=trn #记录前一个场景的trail number
    # 个体间的平均相关系数 pearsonr_allothers_mean
    pearsonr_allothers_mean = (np.sum(pearsonr_list)-np.sum(pearsonr_subji))/(len(pearsonr_list)-len(pearsonr_subji))

    # 判断个体内和个体间平均相关系数的大小
    compare_flag = int(pearsonr_subji_mean>pearsonr_allothers_mean)
    return compare_flag,pearsonr_subji_mean,pearsonr_allothers_mean

# binary classification accuracy
def binary_classification_accuracy(VFC,trn):
    
    compare_flag_list = []
    for trl_idx0 in range(VFC.shape[1]):
        pearsonr_list = []
        for trl_idx1 in range(VFC.shape[1]):
            pearsonr = np.corrcoef(VFC[:,trl_idx0],VFC[:,trl_idx1])[0,1]
            pearsonr_list.append(pearsonr)
        compare_flag,pearsonr_subji_mean,pearsonr_allothers_mean = get_pearsonr_allothers_mean(pearsonr_list,trn,trl_idx0)
        compare_flag_list.append(compare_flag)
        if trl_idx0==0:
            print(pearsonr_subji_mean,pearsonr_allothers_mean)
    
    # 计算正确的比例
    acrcy = np.sum(compare_flag_list)/len(compare_flag_list)

    return acrcy

# topK accuracy with subject_mean pearsonr
def topk_sim_accuracy_meanpearsonr(VFC,trn,k):
    
    # 计算真实标签
    true_labels = get_true_labels(trn)
    true_mean_labels = np.array(range(trn.shape[1]))

    pearsonr_topk_idx_list = []
    for trl_idx0 in range(VFC.shape[1]):
        pearsonr_list = []
        for trl_idx1 in range(VFC.shape[1]):
            pearsonr = np.corrcoef(VFC[:,trl_idx0],VFC[:,trl_idx1])[0,1]
            pearsonr_list.append(pearsonr)
        pearsonr_subj_mean = get_pearsonr_subj_mean(pearsonr_list,trn,trl_idx0)
        pearsonr_topk_idx,pearsonr_topk_value = get_topk_idx(pearsonr_subj_mean,k)
        # print(pearsonr_topk_value)
        if trl_idx0==0:
            pearsonr_topk_idx_list=pearsonr_topk_idx
        else:
            pearsonr_topk_idx_list=np.c_[pearsonr_topk_idx_list,pearsonr_topk_idx]

    # 比较标签，数值为0表示相等
    labels_comp = np.ones(true_labels.shape)
    for i in range(k):
        labels_comp_i = true_mean_labels[list(pearsonr_topk_idx_list[i,:])]-true_labels
        labels_comp *= labels_comp_i

# 计算正确的比例
    acrcy = np.count_nonzero(labels_comp==0)/len(labels_comp)

    return acrcy,pearsonr_topk_idx_list

## show middle results
# topK accuracy with subject_mean pearsonr
def topk_sim_accuracy_meanpearsonr_middle_results(VFC,trn,k):
    
    # 计算真实标签
    true_labels = get_true_labels(trn)
    true_mean_labels = np.array(range(trn.shape[1]))

    pearsonr_topk_idx_list = []
    pearsonr_value_list = []
    for trl_idx0 in range(VFC.shape[1]):
        pearsonr_list = []
        for trl_idx1 in range(VFC.shape[1]):
            if trl_idx1 != trl_idx0:
                pearsonr = np.corrcoef(VFC[:,trl_idx0],VFC[:,trl_idx1])[0,1]
                pearsonr_list.append(pearsonr)
        pearsonr_subj_mean = get_pearsonr_subj_mean(pearsonr_list,trn)
        pearsonr_topk_idx,pearsonr_topk_value = get_topk_idx(pearsonr_subj_mean,k)
        if trl_idx0==0:
            pearsonr_topk_idx_list=pearsonr_topk_idx
            pearsonr_value_list = pearsonr_subj_mean
        else:
            pearsonr_topk_idx_list=np.c_[pearsonr_topk_idx_list,pearsonr_topk_idx]
            pearsonr_value_list = np.c_[pearsonr_value_list,pearsonr_subj_mean]

    # 比较标签，数值为0表示相等
    labels_comp = np.ones(true_labels.shape)
    for i in range(k):
        labels_comp_i = true_mean_labels[list(pearsonr_topk_idx_list[i,:])]-true_labels
        labels_comp *= labels_comp_i

# 计算正确的比例
    acrcy = np.count_nonzero(labels_comp==0)/len(labels_comp)

    return acrcy,pearsonr_topk_idx_list,pearsonr_value_list

