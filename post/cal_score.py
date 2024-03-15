from score import f1sco
from score import recall
from score import F1
from score import Cen_cal
from score import dis_cal
from score import ARI
from score import NMI
from score import Purity
from score import SC
from score import CH

from score import calculate_cluster_centrosize
from score import calc_scoree


import numpy as np



def cal_sc(args,data,label,uniq,gt_table,umap_data,labels):
    # print("data.shape",data.shape)
    # print("label.shape",label.shape)
    # print("pred.shape",pred.shape)
    # print("uniq.shape",uniq.shape)
    # print(data)
    # id_name = np.unique
    label = data.iloc[:,1].values
    pred = data.iloc[:,3].values
    # print("label",label,"shape",label.shape)
    # print("pred",pred,"shape",pred.shape)
    dataa = data.iloc[:,4:].values
    # print("data","shape",dataa.shape)
    # print("uniq","shape",uniq.shape)
    # print("gt_table","shape",gt_table.shape)
    # print("gt_table",gt_table)
    # ch = CH(dataa,pred)
    ch = CH(dataa,data.iloc[:,0].values)
    # ch = CH(umap_data,labels)

    # sc = SC(dataa,pred)
    sc = SC(dataa,data.iloc[:,0].values)
    # sc = SC(umap_data,labels)

    nmi = NMI(label,pred)

    ari = ARI(label,pred)

    calc_sc = calc_scoree(args,dataa,data.iloc[:,0].values,uniq,gt_table)

    cluster_matrix = calculate_cluster_centrosize(dataa, label,'others')
    intra_cluster = np.median(cluster_matrix[:, -1].astype(float))
    inter_cluster = np.std(cluster_matrix[:, 1:-1].astype(float))
    cluster_score = inter_cluster / intra_cluster


    print("ch:",ch)
    print("sc:",sc)
    print("nmi:",nmi)
    print("ari:",ari)
    print("cluster_score:",cluster_score)
    print("calc_sc:",calc_sc)

    return ch,sc,nmi,ari,cluster_score




