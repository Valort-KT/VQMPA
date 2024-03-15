import umap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from os.path import join
from numpy.typing import ArrayLike
from matplotlib import cm

from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.stats import pearsonr
import seaborn as sns

from cal_score import cal_sc

import os



def umap_one(args,embed,labels,label_pred,n_cla):
    reducer = umap.UMAP(n_neighbors=15,n_components=n_cla, min_dist=0.1, metric='euclidean', verbose=False)
    reducer.fit(embed.reshape(embed.shape[0], -1))
    umap_data = reducer.transform(embed.reshape(embed.shape[0], -1))
    umap_join = {"umap":umap_data,"embed":embed,"labels":labels,"label_pred":label_pred}
    np.savez(join(args.savepath_dict["feature_spectra_data"], "umap_join100.npz"),**umap_join)
    return umap_data


def group(args,group_annotation,labels):
    unique_groups = np.unique(group_annotation[:,1])
    label_converted = labels[:].astype(object)
    label_converted[:] = 'others'
    args.group = {}
    for i,gp in enumerate(unique_groups):
        ind = np.isin(labels, group_annotation[group_annotation[:,1] == gp, 0])
        label_converted[ind] = i
        args.group[i] = gp
    return label_converted

def umap3d(args,embed,labels,title,xlabel,ylabel,zlabel,s,alpha,show_legend,group_annotation):
    umap_data = umap_one(args,embed,labels,3)

    colormap = 'tab20_others'

    #label  data
    unique_groups = np.unique(group_annotation[:,1])
    label_converted = labels[:].astype(object)
    label_converted[:] = 'others'
    for gp in unique_groups:
        ind = np.isin(labels, group_annotation[group_annotation[:,1] == gp, 0])
        label_converted[ind] = gp

    #plot
    # print(args)
    savepath = join(args.savepath_dict["umap_figures"], title+'.png')
    if isinstance(colormap, str):
        cmap = cm.get_cmap(colormap.replace('_others', '')).colors
    else:
        cmap = colormap
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    i = 0
    for gp in unique_groups:
        if '_others' in colormap and gp == 'others':
            _c = cm.Greys(25)
        else:
            _c = cmap[i % len(cmap)]
            i += 1
        ind = label_converted == gp
        ax.scatter(
            umap_data[ind, 0],
            umap_data[ind, 1],
            umap_data[ind, 2],
            s=s,
            alpha=alpha,
            c=np.array(_c).reshape(1, -1),
            label=gp,
            zorder=0 if gp == 'others' else len(unique_groups) - i + 1,
        )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.view_init(90, 70)
    hndls, names = ax.get_legend_handles_labels()
    leg = ax.legend(
        hndls,
        names,
        prop={'size': 6},
        bbox_to_anchor=(1, 1),
        loc='upper left',
        ncol=1 + len(names) // 20,
        frameon=False,
    )
    for ll in leg.legendHandles:
        ll._sizes = [6]
        ll.set_alpha(1)
    ax.set_zlabel(zlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    savepath
    if savepath:
        fig.savefig(savepath, dpi=300)
        print(f'Figure saved to {savepath}')
    return fig, ax

def umap2d(args,embed,labels,label_pred,title,xlabel,ylabel,s,alpha,show_legend,group_annotation,pro_n_ind,or_ind,to_df,pro_df):
    umap_data = umap_one(args,embed,labels,label_pred,2)

    

    colormap = 'tab20_others'

    #label  data
    unique_groups = np.unique(group_annotation[:,1])
    label_converted = labels[:].astype(object)
    label_converted[:] = 'others'
    for gp in unique_groups:
        ind = np.isin(labels, group_annotation[group_annotation[:,1] == gp, 0])
        label_converted[ind] = gp
    unique_groupss = np.hstack([unique_groups, 'others'])


    pro_labels = labels[pro_n_ind]
    or_label = pro_labels[:].astype(object)
    or_label[:] = 'others'
    for gp in unique_groups:
        ind = np.isin(pro_labels, group_annotation[group_annotation[:,1] == gp, 0])
        or_label[ind] = gp
    or_unique_groups = unique_groups


    # or_pro = or_ind[pro_n_ind]
    # or_pro = 
    # print("umapdaata.shape",umap_data.shape)
    # print("or_pro.shape",or_pro.shape)
    # print("pro_id.shape",pro_n_ind.shape)
    or_umap_data = umap_data[pro_n_ind]
    # print("or_uamp_data.shape",or_umap_data.shape)
    # print("or_label.shape",or_label.shape)
    # print(or_label)
    # print("group_annotation.shape",group_annotation.shape)
    # print(group_annotation)
    
    # to_or_df = to_df.iloc[pro_n_ind]
    or_umap_df = pd.DataFrame(or_umap_data,columns=['umap1','umap2'])
    # print(or_umap_df.shape)
    # print(to_df.shape)
    or_umap = pd.concat([pro_df,or_umap_df],axis=1)

    cal_sc(args,or_umap,or_label,unique_groupss,group_annotation,umap_data,labels)
    #plot
    plot(args,umap_data,unique_groupss,label_converted,title,xlabel,ylabel,s,alpha,show_legend)
    # plot_grey(args,umap_data,unique_groupss,label_converted,f"{title}+grey",xlabel,ylabel,s,alpha,show_legend)
    # plot_kde(args,umap_data,unique_groupss,label_converted,f"{title}+kde",xlabel,ylabel,s,alpha,show_legend)
    # plot(args,or_umap_data,or_unique_groups,or_label,"pro_n_ind",xlabel,ylabel,s,alpha,show_legend)
    # return fig, ax
    return umap_data



def plot_kde(args,umap_data,unique_groups,label_converted,title,xlabel,ylabel,s,alpha,show_legend):
    colormap = 'tab20_others'
    # xlabel="umap1"
    # ylabel = "umap2"
    # s = 0.4
    # alpha = 0.6


    savepath = join(args.savepath_dict["umap_figures"], title+'.png')
    if isinstance(colormap, str):
        cmap = cm.get_cmap(colormap.replace('_others', '')).colors
    else:
        cmap = colormap
    fig,axe = plt.subplots(4,3,figsize=(32,24),sharex=True,sharey=True)
    i = 0
    for i,ax in enumerate(axe.flatten()):
        if i==2:
            continue
        elif i == 5:
            continue
        gp = unique_groups[i]
        if '_others' in colormap and gp == 'others':
            _c = cm.Greys(50)
            continue
        else:
            _c = cmap[i % len(cmap)]
            i += 1
        ind = label_converted == gp
        sns.kdeplot(x=umap_data[ind, 0],y=umap_data[ind, 1],ax=ax,
                    color=_c,
                    fill = True,
                    # cut=10,
                    # clip=(-10,10),
                    )
        # ax.set_axis_off()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(f"{gp}")
        # ax.xlim = (-10,10)
        # ax.ylim = (-10,10)

    # axe.set(xlim=(-10, 10), ylim=(-10, 10))

    # axe.set(xlim=(-50, 50), ylim=(-10, 5))
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # hndls, names = ax.get_legend_handles_labels()
    # leg = ax.legend(
    #     hndls,
    #     names,
    #     prop={'size': 6},
    #     bbox_to_anchor=(1, 1),
    #     loc='upper left',
    #     ncol=1 + len(names) // 80,
    #     frameon=False,
    # )
    # for ll in leg.legendHandles:
    #     ll._sizes = [6]
    #     ll.set_alpha(1)
    # ax.set_xlabel(xlabel)
    # ax.set_ylabel(ylabel)
    # ax.set_title(title)
    # legend_ax = fig.add_subplot()
    hndls, names = axe.get_legend_handles_labels()
    leg = axe.legend(
        hndls,
        names,
        prop={'size': 6},
        bbox_to_anchor=(1, 1),
        loc='upper left',
        ncol=1 + len(names) // 80,
        frameon=False,
    )
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    """
    您可以使用Figure对象的subplots_adjust()方法来控制子图的大小和间距。该方法接受多个参数，包括left、right、bottom、top、wspace和hspace，用于控制子图在画布上的位置和间距。
    """
    # savepath
    if savepath:
        fig.savefig(savepath, dpi=500,)
        # print(f'Figure saved to {savepath}')




    
def plot_grey(args,umap_data,unique_groups,label_converted,title,xlabel,ylabel,s,alpha,show_legend):
    colormap = 'tab20_others'
    xlabel="umap1"
    ylabel = "umap2"
    s = 0.4
    alpha = 0.6

    #label  data
    # label_converted[:] = 'others'
    # for gp in unique_groups:
    #     ind = np.isin(labels, group_annotation[group_annotation[:,1] == gp, 0])
    #     label_converted[ind] = gp
    # unique_groups = np.hstack([unique_groups, 'others'])
    #plot
    # print(args)
    # path = join(args.savepath_dict["umap_figures"],)
    savepath = join(args.savepath_dict["umap_figures"], title+'.png')
    if isinstance(colormap, str):
        cmap = cm.get_cmap(colormap.replace('_others', '')).colors
    else:
        cmap = colormap
    fig,ax = plt.subplots(1,figsize=(10,10))
    i = 0
    for gp in unique_groups:
        if '_others' in colormap and gp == 'others':
            _c = cm.Greys(50)
        elif gp == 'cytoplasmic':
            _c = cmap[i % len(cmap)]
            i += 1
        elif gp == 'nucleoplasm':
            _c = cmap[i % len(cmap)]
            i += 1
        else:
            _c = cm.Greys(50)
            i += 1
        ind = label_converted == gp
        ax.scatter(
            umap_data[ind, 0],
            umap_data[ind, 1],
            marker='.',
            s=s,
            alpha=alpha,
            c=np.array(_c).reshape(1, -1),
            label=gp,
            zorder=0 if gp == 'others' else len(unique_groups) - i + 1,
        )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
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
    for ll in leg.legendHandles:
        ll._sizes = [6]
        ll.set_alpha(1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    # savepath
    if savepath:
        fig.savefig(savepath, dpi=1000,)
        # print(f'Figure saved to {savepath}')




def plot(args,umap_data,unique_groups,label_converted,title,xlabel,ylabel,s,alpha,show_legend):
    colormap = 'tab20_others'
    xlabel="umap1"
    ylabel = "umap2"
    s = 0.4
    alpha = 0.6

    #label  data
    # label_converted[:] = 'others'
    # for gp in unique_groups:
    #     ind = np.isin(labels, group_annotation[group_annotation[:,1] == gp, 0])
    #     label_converted[ind] = gp
    # unique_groups = np.hstack([unique_groups, 'others'])
    #plot
    # print(args)
    # path = join(args.savepath_dict["umap_figures"],)
    savepath = join(args.savepath_dict["plot_duibi"], title+'.png')
    if isinstance(colormap, str):
        cmap = cm.get_cmap(colormap.replace('_others', '')).colors
    else:
        cmap = colormap
    fig,ax = plt.subplots(1,figsize=(10,10))
    i = 0
    for gp in unique_groups:
        if '_others' in colormap and gp == 'others':
            _c = cm.Greys(25)
        else:
            _c = cmap[i % len(cmap)]
            i += 1
        ind = label_converted == gp
        ax.scatter(
            umap_data[ind, 0],
            umap_data[ind, 1],
            marker='.',
            s=s,
            alpha=alpha,
            c=np.array(_c).reshape(1, -1),
            label=gp,
            zorder=0 if gp == 'others' else len(unique_groups) - i + 1,
        )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
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
    for ll in leg.legendHandles:
        ll._sizes = [6]
        ll.set_alpha(1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    # savepath
    if savepath:
        fig.savefig(savepath, dpi=1000,)
        # print(f'Figure saved to {savepath}')





def plot9(args,umap_data,df,labels,title):
    colormap = 'tab20_others'
    xlabel="umap1"
    ylabel = "umap2"
    s = 0.4
    alpha = 0.6

    #label  data
    unique_groups = np.unique(df.iloc[:,3])
    label_converted = df.iloc[:,3]
    # label_converted[:] = 'others'
    # for gp in unique_groups:
    #     ind = np.isin(labels, group_annotation[group_annotation[:,1] == gp, 0])
    #     label_converted[ind] = gp
    # unique_groups = np.hstack([unique_groups, 'others'])
    #plot
    # print(args)
    # path = join(args.savepath_dict["umap_figures"],)
    savepath = join(args.savepath_dict["plot"], title+'.png')
    if isinstance(colormap, str):
        cmap = cm.get_cmap(colormap.replace('_others', '')).colors
    else:
        cmap = colormap
    fig,ax = plt.subplots(1,figsize=(10,10))
    i = 0
    for gp in unique_groups:
        if '_others' in colormap and gp == 'others':
            _c = cm.Greys(25)
        else:
            _c = cmap[i % len(cmap)]
            i += 1
        ind = label_converted == gp
        ax.scatter(
            umap_data[ind, 0],
            umap_data[ind, 1],
            marker='.',
            s=s,
            alpha=alpha,
            c=np.array(_c).reshape(1, -1),
            label=gp,
            zorder=0 if gp == 'others' else len(unique_groups) - i + 1,
        )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
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
    for ll in leg.legendHandles:
        ll._sizes = [6]
        ll.set_alpha(1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    # savepath
    if savepath:
        fig.savefig(savepath, dpi=1000,)
        print(f'Figure saved to {savepath}')





def umap_init_savepath(args):
    args.savepath_dict["umap_path"]=join(args.savepath_dict['homepath'], 'analysis')
    folders = ['umap_figures', 'umap_data', 'feature_spectra_figures', 'feature_spectra_data',"plot_duibi","plot"]
    for f in folders:
        p = join(args.savepath_dict['umap_path'], f)
        if not os.path.exists(p):
            os.makedirs(p)
        args.savepath_dict[f] = p


def selfpearson_multi(data = None , num_workers = 1):
    """
    Compute self pearson correlation using multiprocessing

    Parameters
    ----------
    data : ArrayLike
        2D Numpy array; self-correlation is performed on axis 1 and iterates over axis 0
    num_workers : int
        Number of workers

    Returns
    -------
    2D Numpy array

    """
    print("Data.shape: ", data.shape)
    print(data)
    print(data[0])
    print(data[0:])
    corr=[]
    for i ,row in enumerate(tqdm(data)):
        co = corr_single(i,row,data[i:],data.shape[0])
        corr.append(co)
    # corr = Parallel(n_jobs=num_workers, prefer='threads')(
    #     delayed(corr_single)(i, row, data[i:], data.shape[0]) for i, row in enumerate(tqdm(data))
    # )

    corr = np.vstack(corr)
    corr_up = np.triu(corr, k=1)
    return corr_up.T + corr


def corr_single(offset: int, array: ArrayLike, matrix: ArrayLike, dims: int) -> ArrayLike:
    """
    Compute pearson's correlation between an array & a matrix

    Parameters
    ----------
    offset : int
        Offset
    array : ArrayLike
        1D Numpy array
    matrix : ArrayLike
        Numpy array
    dims : int
        Shape size in the second dimension of the correlation

    Returns
    -------
    Numpy array

    """
    corr = np.zeros((1, dims))
    for ii, row in enumerate(matrix):
        corr[:, ii + offset] = pearsonr(np.nan_to_num(array,nan=0.00001), np.nan_to_num(row,nan=0.00001))[0]
    return corr


def pearson_multi(array: ArrayLike, matrix: ArrayLike, num_workers: int = 1) -> ArrayLike:
    """
    Compute pearson's correlation between an array & a matrix

    Parameters
    ----------
    array : ArrayLike
        1D Numpy array
    matrix : ArrayLike
        2D Numpy matrix
    num_workers : int
        Number of workers

    Returns
    -------
    Numpy array

    """
    if array.shape != matrix.shape[1:]:
        raise ValueError('array.shape must equal matrix.shape[1:].')
    corr = Parallel(n_jobs=num_workers)(delayed(pearsonr)(array, ar) for ar in tqdm(matrix))
    return np.vstack(corr)[:, 0]



def calculate_corr(data = None,num_workers=1,args = None):
    """
        Compute self pearson's correlation between vq index and vq index

        Parameters
        ----------
        data : ArrayLike
            Numpy array with VQ index on dim 1
        num_workers : int
            Number of workers
        filepath : str
            File path (including file name & extension)

        Returns
        -------
        Numpy array

        """
    print('Computing self Pearson correlation...')
    corr_idx = np.nan_to_num(selfpearson_multi(data.T, num_workers=num_workers))
    print('Computing Pearson correlation between VQ index and VQ index...')
    if args.OTHER:
        filepath = join(args.OTHER, 'corr_idx.npy')
        np.save(filepath, corr_idx)
    return corr_idx 

def plot_clustermap(
        args,
        cell_data = None,
        vq_idx=1,
        num_workers=1,
        name=None,
        update_feature = True,
):
    """
        Generate hierarchical clustering heatmaps against vqind vs. vqind
        Parameters
        ----------
        vq_idx : int
            VQ layer index
        data_loader : DataLoader
            DataLoader to compute the matrix
        num_workers : int
            Number of workers
        filepath : str
            File path (including file name & extension)
        use_codebook : bool
            Uses codebook to compute self-correlation in VQ indices if Ture, otherwise uses cell id
        update_feature_spectrum_indices : bool
            Overwrite class attribute update_feature_spectrum_indices if True

        Returns
        -------
        Seaborn heatmap object

    """

    _mat_idx = cell_data
    # _mat_idx = np.nan_to_num(_mat_idx, nan=0.00001)
    corr_idx = calculate_corr(_mat_idx, num_workers=num_workers, args = args)
    print(corr_idx.shape)
    print('Plotting clustermap...')

    heatmap = sns.clustermap(
        corr_idx,
        cmap='RdBu_r',
        vmin=-1,
        vmax=1,
        cbar_kws=None,
        cbar_pos=None,
        # col_cluster=False,
        # row_linkage=None,
    )

    heatmap.ax_col_dendrogram.set_title(f'vq{vq_idx} indhist Pearson corr hierarchy link')
    heatmap.ax_heatmap.set_xlabel('vq index')
    heatmap.ax_heatmap.set_ylabel('vq index')
    if update_feature:
        feature_spectrum_indices = np.array(heatmap.dendrogram_row.reordered_ind)


    filepath = join(args.OTHER, f'clustermap_vq{vq_idx}_{name}.png')
    heatmap.savefig(filepath, dpi=300)
    return heatmap


def computeee_feature_spectrum(
        data_loader = None,
        num_workers=1,
        use_codebook = False,
        update_feature = True,
):
    """
        Compute feature spectrum

        Parameters
        ----------
        data_loader : DataLoader
            DataLoader to compute the matrix
        num_workers : int
            Number of workers
        filepath : str
            File path (including file name & extension)
        use_codebook : bool
            Uses codebook to compute self-correlation in VQ indices if Ture, otherwise uses cell id
        update_feature_spectrum_indices : bool
            Overwrite class attribute update_feature_spectrum_indices if True

        Returns
        -------
        Numpy array

    """
    if use_codebook:
        _mat_idx = data_loader.codebook
    else:
        _mat_idx = data_loader.cell_id
    corr_idx = calculate_corr(_mat_idx, num_workers=num_workers)
    if update_feature:
        feature_spectrum_indices = np.argsort(corr_idx)[::-1]
    return corr_idx


def compute_feature_spectrum(args,
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


def plot_feature_spectrum(args,
                          cell_idx,
                          trus=None,
                          cla_re = None,
                          feature_spec_idx=None,
                          name=None,
                          update_feature = True,
                          ):
    """
        Plot feature spectrum

        Parameters
        ----------
        cell_idx : ArrayLike
            Numpy array with cell id on dim 1
        feature_spec_idx : ArrayLike
            Numpy array with feature spectrum indices
        name : str
            Name of the figure
        update_feature_spectrum_indices : bool
            Overwrite class attribute update_feature_spectrum_indices if True

        Returns
        -------
        Seaborn heatmap object

    """
    ft_spec = compute_feature_spectrum(args, cell_idx, feature_spec_idx, name)
    print('Plotting feature spectrum...')

    x_max = ft_spec[0].shape[1]+1
    x_ticks = np.arange(0, x_max, 50)
    fig,ax = plt.subplots(figsize=(10, 3))
    print(ft_spec[0].shape)
    ax.stairs(ft_spec[0][0], np.arange(x_max), fill=True)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlabel('Feature index')
    ax.set_ylabel('Counts')
    ax.set_xlim([0, x_max])
    ax.set_xticks(x_ticks, ft_spec[1][x_ticks])
    fig.tight_layout()
    fig.show()
    fig.savefig(join(args.OTHER, f'feature_spectrum_{name}_0.png'), dpi=300)




# def label_cla(args,label_true,label_pred,gt_label,mode=None):
#     ind_true =np.isin(label_true[:],gt_label[:,0])
#     ind_pred = np.isin(label_pred[:],gt_label[:,0])
#     print(np.sum(ind_true==True),np.sum(ind_pred==True))
#     or_ind = ((ind_true==True) | (ind_pred==True))
#     ind = ((ind_true==True) & (ind_pred==True))
#     # ind  所有预测正确的
#     ind_true_nopred = ((ind_true==True) & (ind==False))

#     #  ind_true_nopred  原值中没有被预测到的
#     ind_pred_notrue = ((ind_pred==True) & (ind==False))
#     #  ind_pred_notrue  预测值中没有被原值包含的


#     or_ind_notrue = ((ind_true==False) & (or_ind==True))

#     or_ind_nopred = ((ind_pred==False) & (or_ind==True))

#     print("ind:",np.sum(ind==True),"ind_true_nopred:",np.sum(ind_true_nopred==True),"ind_pred_notrue:",np.sum(ind_pred_notrue==True))
#     print("or_ind:",np.sum(or_ind==True),"or_ind_notrue:",np.sum(or_ind_notrue==True),"or_ind_nopred:",np.sum(or_ind_nopred==True))

#     or_true = label_true[or_ind]
#     or_pred = label_pred[or_ind]
#     or_gt_true=[]
#     or_gt_pred=[]
#     m,n=0,0
#     for i in range(len(or_true)):
#         ind_tr = np.isin(gt_label[:,0],or_true[i])
#         if np.any(ind_tr==True):
#             m = m+1
#             or_gt_true.append(gt_label[ind_tr,1][0])
#         else:
#             or_gt_true.append("other")
#         # or_gt_true.append(gt_label[ind_tr,1][0])
#         ind_pr = np.isin(gt_label[:,0],or_pred[i])
#         if np.any(ind_pr==True):
#             n = n+1
#             or_gt_pred.append(gt_label[ind_pr,1][0])
#         else:
#             or_gt_pred.append("other")
#     # print(or_gt_true,or_gt_pred)
#     or_gt_true = np.array(or_gt_true)
#     or_gt_pred = np.array(or_gt_pred)
#     print("m,n",m,n)
#     to_df = np.hstack((or_true.reshape(-1,1),or_gt_true.reshape(-1,1),or_pred.reshape(-1,1),or_gt_pred.reshape(-1,1)))
#     print(to_df.shape,to_df)
#     j=0
#     for i in range(len(or_ind)):
#         if or_gt_true[i] == or_gt_pred[i]:
#             j = j+1
#     print(j,i)


#     or_gt1_true=[]
#     or_gt1_pred=[]
#     ind_all=[]
#     m,n=0,0
#     for i,ix in enumerate(or_ind):
#         ind_tr = np.isin(gt_label[:,0],label_true[i])
#         if np.any(ind_tr==True):
#             m = m+1
#             or_gt1_true.append(gt_label[ind_tr,1][0])
#         else:
#             or_gt1_true.append("other")
#         # or_gt_true.append(gt_label[ind_tr,1][0])
#         ind_pr = np.isin(gt_label[:,0],or_pred[i])
#         if np.any(ind_pr==True):
#             n = n+1
#             or_gt1_pred.append(gt_label[ind_pr,1][0])
#         else:
#             or_gt1_pred.append("other")

#     # or   包含所有原值和预测值
#     if mode=="or":
#         return or_true,or_pred
#     # yu 同时等于gt——label的
#     elif mode=="yu":
#         return label_true[ind],label_pred[ind]
#     # equa  各细胞器直接相等的
#     elif mode=="equa":

# # print(args)
#     savepath = join(args.savepath_dict["umap_figures"], title+'.png')
#     if isinstance(colormap, str):
#         cmap = cm.get_cmap(colormap.replace('_others', '')).colors
#     else:
#         cmap = colormap
#     fig,ax = plt.subplots(1,figsize=(8,8))
#     i = 0
#     for gp in unique_groups:
#         if '_others' in colormap and gp == 'others':
#             _c = cm.Greys(25)
#         else:
#             _c = cmap[i % len(cmap)]
#             i += 1
#         ind = label_converted == gp
#         ax.scatter(
#             umap_data[ind, 0],
#             umap_data[ind, 1],
#             marker='.',
#             s=s,
#             alpha=alpha,
#             c=np.array(_c).reshape(1, -1),
#             label=gp,
#             zorder=0 if gp == 'others' else len(unique_groups) - i + 1,
#         )
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     hndls, names = ax.get_legend_handles_labels()
#     leg = ax.legend(
#         hndls,
#         names,
#         prop={'size': 6},
#         bbox_to_anchor=(1, 1),
#         loc='upper left',
#         ncol=1 + len(names) // 20,
#         frameon=False,
#     )
#     for ll in leg.legendHandles:
#         ll._sizes = [6]
#         ll.set_alpha(1)
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     ax.set_title(title)
#     fig.tight_layout()
#     # savepath
#     if savepath:
#         fig.savefig(savepath, dpi=1000,)
#         print(f'Figure saved to {savepath}')

