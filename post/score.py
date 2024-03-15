import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import umap
import pandas as pd
import math
from sklearn import metrics
from tqdm import tqdm
from sklearn.cluster import KMeans
# from DBI import evalution


from typing import Sequence, Optional

import numpy as np
from numpy.typing import ArrayLike


def calc_scoree(args,
            data=None,
            label = None,
            # gt_name = None,
            uniq_group=None,
            gt_table=None,
            # titles = None,
            # calc_mode = None,
            # savepath = None,
    ):
    gt_name = gt_table[:,0]
    uniq = np.unique(gt_table[:,1])
    data = np.sum(data,axis=1)
    # print(label)
    ind = np.isin(label[:], gt_name)
    df = pd.DataFrame(data[ind],columns=['val'])
    df.insert(df.shape[1],'cate','others')
    # print("df",df)
    for i, fname in enumerate(tqdm(uniq)):
        group0 = gt_table[gt_table[:, 1] == fname]
        ind = np.isin(label[:], group0[:, 0])
        df1 = pd.DataFrame(data[ind],columns=['val'])
        df1.insert(df1.shape[1],'cate',fname)
        df = pd.concat((df,df1),axis=0)
    # label = np.append(uniq_group,'others')
    # print("df",df)
    # print(f"Calc clumster score,nums:{len(data)}")
    umap_score = score(data=df,label=uniq)
    # with open('/data/hkt/VAE/hkt/model_outputs/umaps/umap_score',mode='a+') as f:
    return umap_score
        # print(savepath)
        # path = os.path.join(savepath,'umap_score.txt')
        # with open(path,mode='a+') as f:
        #     f.write(self.time)
        #     f.write(f"Calc clumster score,nums:{len(data)}. {titles}:{calc_mode} is : {umap_score}\n")
        #     f.close

# print("umap_score")
def score(
            data=None,label = None,
    ):
        set_mean=[]
        set_devition=[]
        # print(label)
        # df = pd.DataFrame(data,columns=['val'])
        # df = pd.concat((df,pd.DataFrame(label,columns=['cate'])),axis=1)
        
        for i,name in enumerate(label):
            # print(data['cate']==name)
            da = data[data['cate']==name]
            da_np = np.array(da['val'])
            # da_np = np.sum(da_np,axis=1)
            median = np.median(da_np)
            std = np.sqrt(np.sum((da_np-median)**2)/len(da_np))
            set_mean.append(median)
            set_devition.append(std)
        median = np.median(set_devition)
        std = np.sqrt(np.sum((set_mean-np.median(set_mean))**2)/len(set_mean))

        return std/median


# def calculate_cluster_centrosize(data: ArrayLike, cluster_label: ArrayLike, exclude: Optional[Sequence] = None):
#     """
#     Generates a matrix with different clusters on dimension 0
#     and cluster name, centroid coordinates & cluster size on dimension 1.
#     It will be used to compute the cluster score for a given UMAP

#     Parameters
#     ----------
#     data : ArrayLike
#         UMAP data
#     cluster_label : ArrayLike
#         Numpy array labeling each cluster; must be same length as data
#     exclude : str or Sequence
#         Labels to exclude from calculating cluster sizes

#     Returns
#     -------
#     Numpy array with cluster_name, centroid, cluster_size on each column

#     """
#     cluster_uniq = np.unique(cluster_label)
#     if exclude is not None:
#         cluster_uniq = cluster_uniq[~np.isin(cluster_uniq, exclude)]

#     centroid_list = []
#     clustersize_list = []
#     for cl in cluster_uniq:
#         ind = cluster_label == cl
#         data0 = data[ind.flatten()]
#         centroid = np.median(data0, axis=0)
#         # square distance between each datapoint and centroid
#         square_distance = (centroid - data0) ** 2
#         # median of sqrt of square_distance as cluster size
#         cluster_size = np.median(np.sqrt(square_distance[:, 0] + square_distance[:, 1]))
#         centroid_list.append(centroid)
#         clustersize_list.append(cluster_size)
#     cluster_matrix = np.vstack([cluster_uniq, np.vstack(centroid_list).T, clustersize_list]).T
#     # cluster_matrix = calculate_cluster_centrosize(umap_data, label_data, 'others' if '_others' in colormap else None)
#     intra_cluster = np.median(cluster_matrix[:, -1].astype(float))
#     inter_cluster = np.std(cluster_matrix[:, 1:-1].astype(float))
#     cluster_score = inter_cluster / intra_cluster
#     return cluster_score






"""
label:每个蛋白质对应的定位信息
pred：预测值
data: umap降维后的数据

"""



"""
调整兰德系数： ARI（Adjusted Rand Index）
该指标度量聚类结果与真实标签之间的相似度，
值范围在-1到1之间，值越大表示聚类结果与真实标签越相似。
"""

def ARI(label,pred):
    return metrics.adjusted_rand_score(label,pred)



"""
Normalized Mutual Information（NMI）：
该指标度量聚类结果与真实标签之间的互信息，
值范围在0到1之间，值越大表示聚类结果与真实标签越相似
"""

def NMI(label,pred):
    return metrics.normalized_mutual_info_score(label,pred)


"""
Fowlkes-Mallows Index（FMI）：
该指标度量聚类结果与真实标签之间的精度和召回率，
值范围在0到1之间，值越大表示聚类结果与真实标签越相似。
"""

def FMI(label,pred):
    return metrics.fowlkes_mallows_score(label,pred)

"""
Purity
因为聚类纯度的总体思想也用聚类正确的样本数除以总的样本数，
因此它也经常被称为聚类的准确率。
"""

def Purity(label,pred,uniq):
    sum = 0
    pred = np.reshape(pred,(-1,1))
    for i,da in enumerate(uniq):
        ind = np.isin(label[:],da)
        p = pred[ind,:]
        a = np.sum(p == da)
        sum +=a
    return float(sum/len(label))

"""
Silhouette Coefficient（轮廓系数）：
计算每个数据点到其所属簇和最近的邻居簇的平均距离，
以此度量聚类的紧密度和分离度。
该指标的取值范围在[-1, 1]之间，越接近1表示聚类效果越好。
"""

def SC(data,pred):
    return metrics.silhouette_score(data,pred)

"""
Calinski-Harabasz Index：
计算簇间平均方差与簇内平均方差的比值，衡量簇的分离度。
"""
def CH(data,pred):
    return metrics.calinski_harabasz_score(data,pred)

"""
SSE（Sum of Squared Errors）
该统计参数计算的是拟合数据和原始数据对应点的误差的平方和
SSE越接近于0，说明模型选择和拟合更好，数据预测也越成功。
"""
def dis_cal(cen,all):
    sum=0
    for i in range(len(all)):
        number = math.sqrt((all[i][0]-cen[0])**2+(all[i][1]-cen[1])**2)
        sum += number**2
    return sum


#质心计算::Centroid calculation
def Cen_cal(uniq,label,data):
    da = {
        # 'group':uniq_group,
        'x':[],
        'y':[],
        'sum':[],}
    
    for i in uniq:
        ind = np.isin(label[:],i)
        embedding = data[ind,:]
        center = np.mean(embedding,axis=0)
        da['x'].append(center[0])
        da['y'].append(center[1])
        da['sum'].append(dis_cal(center,embedding))

    df = pd.DataFrame(da,index=uniq)
    print(df)
    sum = df['sum'].sum()

    return sum


"""
F1corse
F-score 就是分别从两个角度，
主观（Predicted）和客观（Actual）上去综合的分析TP够不够大。
这也就是我们平常看到的结论 F-score的值 
只有在Precision 和 Recall 都大的时候 才会大。
"""


def F1(label,pred,beta=1.):
    (tn, fp), (fn, tp) = metrics.cluster.pair_confusion_matrix(label, pred)
    ri = (tp + tn) / (tp + tn + fp + fn)
    ari = 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    p, r = tp / (tp + fp), tp / (tp + fn)
    f_beta = (1 + beta**2) * (p * r / ((beta ** 2) * p + r))
    return f_beta


def recall(label,pred):
    tp = sum([yt == yp == 1 for yt, yp in zip(label, pred)])
    fn = sum([yt == 1 and yp == 0 for yt, yp in zip(label, pred)])
    return tp / (tp + fn)


def f1sco(label,pred):
    f1_macro = metrics.f1_score(label, pred, average='macro')
    return f1_macro



def calculate_cluster_centrosize(data: ArrayLike, cluster_label: ArrayLike, exclude: Optional[Sequence] = None):
    """
    Generates a matrix with different clusters on dimension 0
    and cluster name, centroid coordinates & cluster size on dimension 1.
    It will be used to compute the cluster score for a given UMAP

    Parameters
    ----------
    data : ArrayLike
        UMAP data
    cluster_label : ArrayLike
        Numpy array labeling each cluster; must be same length as data
    exclude : str or Sequence
        Labels to exclude from calculating cluster sizes

    Returns
    -------
    Numpy array with cluster_name, centroid, cluster_size on each column

    """
    cluster_uniq = np.unique(cluster_label)
    if exclude is not None:
        cluster_uniq = cluster_uniq[~np.isin(cluster_uniq, exclude)]

    centroid_list = []
    clustersize_list = []
    for cl in cluster_uniq:
        ind = cluster_label == cl
        data0 = data[ind.flatten()]
        centroid = np.median(data0, axis=0)
        # square distance between each datapoint and centroid
        square_distance = (centroid - data0) ** 2
        # median of sqrt of square_distance as cluster size
        cluster_size = np.median(np.sqrt(square_distance[:, 0] + square_distance[:, 1]))
        centroid_list.append(centroid)
        clustersize_list.append(cluster_size)
    return np.vstack([cluster_uniq, np.vstack(centroid_list).T, clustersize_list]).T



