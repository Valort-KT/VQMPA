
import math
import os
import subprocess
import sys
import time
import argparse
import datetime
import numpy as np
import random
import umap
from collections import Counter

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms.functional as visionF
import torch.cuda.amp as amp
from typing import Optional
import matplotlib.pyplot as plt

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
from timm.utils import ModelEma as ModelEma
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from configs import *
from models import *
# from loss import *
from data import *
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import *
from post.umap_utils import umap2d,umap_init_savepath,umap3d,umap_one,group,plot_clustermap,plot_feature_spectrum


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_option():


    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--data',type=str,default='two_plas',help='dataset')
    # parser.add_argument('--cfg', type=str, default="/data/hkt/work_microplastic/res_50_two/configs/microplas/res50_two.yaml" ,metavar="FILE", help='path to config file', )#required=True,
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, default=8, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, default='/data/hkt/work_microplastic/data_pre/sam_six_other', help='path to dataset')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--finetune', help='finetune from checkpoint')

    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")

    parser.add_argument('--output', default='/data/hkt/work_microplastic/result/', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', default=False,action='store_true', help='Perform evaluation only')

    # ema
    parser.add_argument('--model-ema',default=False, action='store_true')


    # distributed training
    parser.add_argument("--local_rank", type=int, required=False, help='local rank for DistributedDataParallel')

    parser.add_argument('--dist-url', default='tcp://127.0.0.1:8090', type=str,
                    help='url used to set up distributed training')

    args, unparsed = parser.parse_known_args()
    config = select_config(args)

    return args, config


def load_data_model(config,):
    model = build_model(config)
    model.cuda()
    check_path = os.path.join(config.OUTPUT, f'checkpoints/ckpt.pth') 
    checkpoint = torch.load(check_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    dataset_val,data_loader_val = buildval_loader(config)
    class_dict = dataset_val.class_to_idx
    clas_rev = {v:k for k,v in class_dict.items()}
    model.eval()
    vqts = []
    fcts = []
    trus = []
    v2inds = []
    with torch.no_grad():
        for idx,(images,target,_) in enumerate(data_loader_val):
            images = images.cuda()
            # target = target.cuda()
            vqt,fct,v2ind = model(images,mode='vqt')
            vqts.append(vqt.cpu().numpy())
            fcts.append(fct.cpu().numpy())
            trus.append(target.numpy())
            v2inds.append(v2ind.cpu().numpy())
    vqts = np.concatenate(vqts,axis=0)
    fcts = np.concatenate(fcts,axis=0)
    max_id = np.argmax(fcts,axis=1)
    trus = np.concatenate(trus,axis=0)
    v2inds = np.concatenate(v2inds,axis=0)

    return vqts,max_id,trus,clas_rev,v2inds,fcts

from post.score import *
_,config = parse_option()
cudnn.benchmark = True
path_dir = os.path.join(config.OUTPUT, f'checkpoints')
# path_dir = sorted(path_dir, key=lambda x:int(x.split('_')[-1].split('.')[0]))
name_list = os.listdir(path_dir)
name_id = [x.split('_')[-1].split('.')[0] for x in name_list]
name_id = sorted(name_id, key=lambda x:int(x))
colormap = "tab10"
idxx=390
os.makedirs(config.OUTPUT, exist_ok=True)
# world_size = torch.cuda.device_count()
metric_df = pd.DataFrame(columns=["F1","ACC","CH","SC","NMI","ARI","CLUSTER_SC"])
for i in range(2):
    vqt,fct,trus,cla_re,v2_ind,soft = load_data_model(config)
    vqt = vqt.reshape(vqt.shape[0],-1)
    ump = umap.UMAP(n_neighbors=6,min_dist=0.2,metric='euclidean')
    umapda = ump.fit_transform(vqt)
    ch = CH(umapda,trus)
    sc = SC(umapda,trus)
    cluster_matrix = calculate_cluster_centrosize(umapda,trus)
    intra_cluster = np.median(cluster_matrix[:, -1].astype(float))
    inter_cluster = np.std(cluster_matrix[:, 1:-1].astype(float))
    cluster_score = inter_cluster / intra_cluster

    nmi = NMI(trus,fct)
    ari = ARI(trus,fct)
    acc = (trus==fct).sum()/len(trus)
    f1 = f1sco(trus,fct)
    metric_df.loc[i] = [f1,acc,ch,sc,nmi,ari,cluster_score]

count = Counter(trus)
count = {k: v for k, v in sorted(count.items(), key=lambda item: item[0])}
plot_name = trus
config.OTHER = os.path.join(config.OUTPUT, f'postvqt/end')
umap_path = os.path.join(config.OUTPUT, f'postvqt/end/fctumap.png')
metric_df.to_csv(os.path.join(config.OUTPUT, f'postvqt/end/metric.csv'))
if not os.path.exists(config.OTHER):
    os.makedirs(config.OTHER)
uniq_plot= np.unique(plot_name)
fig,ax = plt.subplots(figsize=(5,5))
marker_list = ['o','v','s','d','*','+','x','D','d','|','_']
for k,gp in enumerate(uniq_plot):
    _c = plt.get_cmap(colormap)(k)
    ind = plot_name==gp
    ax.scatter(
        umapda[ind, 0],
        umapda[ind, 1],
        marker=marker_list[k],
        s=50,
        alpha=0.7,
        c=np.array(_c).reshape(1, -1),
        label=cla_re[gp],
    )
hndls, names = ax.get_legend_handles_labels()
leg = ax.legend(
    hndls,
    names,
    prop={'size': 6},
    bbox_to_anchor=(1, 1),
    loc='upper left',
    ncol=1 + len(names) // 80,
    frameon=False,
    )
plt.tight_layout()
plt.show()
fig.savefig(umap_path,dpi=500,bbox_inches='tight')
plt.close()

num_cols = v2_ind.shape[1]
all_zero_ind = []
bool_ind = np.ones(num_cols,dtype=bool)
for col in range(num_cols):
    if np.sum(v2_ind[:,col])==0:
        all_zero_ind.append(col)
        bool_ind[col] = False
    if np.sum(v2_ind[:,col])<20:
        all_zero_ind.append(col)
        bool_ind[col] = False
v2_ind = v2_ind[:,bool_ind]

heatmap = plot_clustermap(config,v2_ind,num_workers=8,name="heatmap_30")

def compute_feature_spectrum(
                             cell_idx,
                             feature_spec_idx=None,
                             name=None,):
    feature_spec_idx = np.array(feature_spec_idx.dendrogram_row.reordered_ind)
    if len(feature_spec_idx.shape) !=1:
        raise ValueError('feature_spectrum_indices must be a 1D array.')
    
    if cell_idx.shape[-1] != len(feature_spec_idx):
            raise ValueError(
                f'The second dim of vq_index_histogram ({cell_idx.shape[-1]}) '
                f'must be same as the length of feature_spectrum_indices ({len(feature_spec_idx)}).'
            )
    
    return cell_idx[:, feature_spec_idx], feature_spec_idx


import seaborn as sns
import pandas as pd

ft_spec = compute_feature_spectrum(v2_ind, feature_spec_idx=heatmap)
ft = ft_spec[0]
ft_x = ft_spec[1]
da_sta = pd.DataFrame(columns=["spec","label","ind"])
for j in range(7):
    la = cla_re[j]
    nui_ind = np.where(trus==j)
    nui = np.sum(ft[nui_ind],axis=0)
    norm_nui = (nui-np.min(nui))/(np.max(nui)-np.min(nui))
    nu_df = pd.DataFrame({"spec":norm_nui,"label":la,"ind":ft_x})
    da_sta = pd.concat([da_sta,nu_df],ignore_index=True)
print('Plotting feature spectrum...')
pal = sns.cubehelix_palette(10, rot=-.25, light=.7)

grid = sns.FacetGrid(da_sta,row="label", hue="label",aspect = 12)
grid.map(plt.stairs,"spec",edges=range(ft.shape[1]+1),fill=True)
grid.despine(left=True, bottom=True)
grid.set_xticklabels([])
grid.set_yticklabels([])
grid.set_titles("")

# 移除刻度线
for ax in grid.axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])



def labelx(x,color,label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,fontsize=50,
            ha="left", va="center", transform=ax.transAxes)
    
grid.map(labelx,"spec")

grid.savefig(os.path.join(config.OTHER, f'epoch_{idxx}_fctspec.png'),dpi=500,bbox_inches='tight')



color = {"color": [
                    "rgba(31, 119, 180, 0.8)",
                    "rgba(255, 127, 14, 0.8)",
                    "rgba(44, 160, 44, 0.8)",
                    "rgba(214, 39, 40, 0.8)",
                    "rgba(148, 103, 189, 0.8)",
                    "rgba(140, 86, 75, 0.8)",
                    "rgba(227, 119, 194, 0.8)",
                    "rgba(127, 127, 127, 0.8)",
                    "rgba(188, 189, 34, 0.8)",
                    "rgba(23, 190, 207, 0.8)",
                    "rgba(174, 199, 232, 0.8)",
                    "rgba(255, 187, 120, 0.8)",
                    "rgba(152, 223, 138, 0.8)",
                    ]
                    }
sn_labe = [v for k,v in cla_re.items()]
fcla = [6,2,5,1,4,3,0]
fcla1 = [cla_re[i] for i in fcla]
sn_label1 = np.array(sn_labe)
sn_label = np.concatenate((sn_labe,sn_label1))
trus_la = []
for i in range(len(trus)):
    trus_la.append(sn_label[trus[i]])
trus_la = np.array(trus_la)
sn_label_color = []
dd = pd.DataFrame(columns=["source","target","value","color"])
for i in range(len(sn_label)):
    m=i%7
    sn_label_color.append(color["color"][m])

for i,la in enumerate(sn_label):
    if i>6:
        break
    for j,lb in enumerate(fcla):
        if j>6:
            continue
        ind = ((trus==i)&(fct==lb))
        x = sn_label_color[i].replace("0.8", str(0.3))
        dd.loc[i*len(sn_label)+j] = [i,j+7,np.sum(ind==True),x]




import plotly.graph_objects as go
import plotly.io as pio
sn_label_2 = sn_label


fig = go.Figure(data=[go.Sankey(
    node = dict(
        pad = 15,
        thickness = 20,
        label = sn_label_2,
        color = sn_label_color,
        
    ),
    textfont=dict(
            size=30,
        ),
    link = dict(
        source = dd['source'].tolist(),
        target = dd['target'].tolist(),
        value = dd['value'].tolist(),
        color = dd['color'].tolist(),
    ))])


fig.show()
path = os.path.join(config.OTHER,"sankey80.png")
pio.write_image(fig, path, width=1000, height=1000, scale=3.5)


import seaborn as sns
# import matplotlib.colormap as cm
from matplotlib import cm
from matplotlib.gridspec import GridSpec
def plot_kde(args,umap_data,unique_groups,label_converted,title,):
    colormap = 'tab10'
    if isinstance(colormap, str):
        cmap = cm.get_cmap(colormap.replace('_others', '')).colors
    else:
        cmap = colormap

    fig = plt.figure(figsize=(24,12))
    gs = GridSpec(2,4, figure=fig)
   
    i = 0
    for j in range(8):
        if j==3:
            continue
        cow = j//4
        row = j%4
        gp = unique_groups[i]
        if '_others' in colormap and gp == 'others':
            _c = cm.Greys(50)
            continue
        else:
            _c = cmap[i % len(cmap)]
            i += 1
        ind = label_converted == gp
        
        ax = fig.add_subplot(gs[cow,row],aspect='equal')
        sns.kdeplot(x=umap_data[ind, 0],y=umap_data[ind, 1],ax=ax,
                    color=_c,
                    fill = True,
                    )
        
        ax.set_xlim(-10,20)
        ax.set_ylim(-10,15)

        # 获取子图的坐标范围
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        # 填充整个图片区域
        plt.fill([x_min, x_max, x_max, x_min], [y_min, y_min, y_max, y_max], color=_c, alpha=0.1)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)  
            
        ax.set_title(f"{gp}",fontdict={"fontsize":25,})

    ax_merged = fig.add_subplot(gs[0,3],aspect='equal')


    uniq = unique_groups
    handls = [plt.Rectangle((0,0),1,1,color=cmap[i % len(cmap)]) for i in range(len(uniq))]

    ax_merged.legend(handls,uniq,prop={'size':30},loc='center')
    ax_merged.axis("off")
    
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    fig.tight_layout()
    """
    您可以使用Figure对象的subplots_adjust()方法来控制子图的大小和间距。该方法接受多个参数，包括left、right、bottom、top、wspace和hspace，用于控制子图在画布上的位置和间距。
    """
    fig.savefig(os.path.join(config.OTHER,"kde.png"), dpi=500,)

plot_kde(config,umapda,sn_labe,trus_la,"umap_kde")


