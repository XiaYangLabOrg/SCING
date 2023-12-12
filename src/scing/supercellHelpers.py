import scanpy as sc
import pandas as pd
import numpy as np
from collections import Counter


def preprocess_data(sub_adata, n_genes=2000, npcs=40, percent_cells=0.7):
    """
    :param sub_adata: scanpy object of cell types
    :param n_genes: number of variable genes to use
    :param npcs: number of pcs to use for preprocessing
    :return: ad_sub: preprocessed data with log1 transformed count data
    """

    sc.pp.normalize_total(sub_adata, target_sum=1e4)
    sc.pp.log1p(sub_adata)
    sc.pp.highly_variable_genes(
        sub_adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=n_genes)
    sub_adata = sub_adata[:, sub_adata.var.highly_variable]

    ad_sub = sc.AnnData(sub_adata.X, obs=sub_adata.obs, var=sub_adata.var)

    saved_log_mat = ad_sub.X.copy()

    sc.pp.scale(ad_sub)
    sc.tl.pca(ad_sub, svd_solver='arpack', n_comps=min(npcs, sub_adata.shape[0] - 1))

    sc.pp.neighbors(ad_sub, n_neighbors=10, n_pcs=min(npcs, sub_adata.shape[0] - 1))

    ad_sub.X = saved_log_mat

    return ad_sub


def merge_cells(ads, variable_to_merge='leiden'):
    """
    :param ads: adata of cells to be merged.
    :param variable_to_merge: observation variable to merge data by (default: leiden)
    :return: merged_data: adata of merged cells
    """
    clust_assignment = ads.obs.loc[:, variable_to_merge].to_numpy()
    clusts = np.unique(clust_assignment)
    ncells = len(clusts)
    new_mat = np.zeros((ncells, ads.shape[1]))
    for i, c in enumerate(clusts):
        locs = np.where(clust_assignment == c)[0]
        new_mat[i, :] = np.mean(ads.X[locs, :], axis=0)
    merged_cells = sc.AnnData(new_mat, var=ads.var)
    return merged_cells


def get_leiden_based_on_ncell(ad_sub, num_cells, verbose):
    """
    :param ad_sub: adata of single celltype
    :param resolutions: np vector of resolutions to test
    :param num_cells: average number of cells per cluster
    :param verbose: Whether or not to print along the way
    :return: ad_sub: adata of single celltype with clusters in leiden
    """
    resolutions = np.arange(0.1, 1000, 0.1)
    vec_length = len(resolutions)
    iter_ = int(vec_length / 2)
    last_iter = 0

    # binary search to find optimal resolution
    while True:
        vec_length = abs(iter_ - last_iter)
        sc.tl.leiden(ad_sub, resolution=resolutions[iter_])

        if abs(iter_ - last_iter) <= 1:
            break
	
        ncells_in_merged = len(np.unique(ad_sub.obs.leiden.to_numpy().ravel()))

        last_iter = iter_

        if ncells_in_merged < num_cells:
            iter_ += int(vec_length / 2)
        elif ncells_in_merged >= num_cells:
            iter_ -= int(vec_length / 2)
	
    # final leiden clustering
    sc.tl.leiden(ad_sub, resolution=resolutions[iter_])

    if verbose:
        print('There are ',
              len(np.unique(ad_sub.obs.leiden.to_numpy().ravel())),
              ' supercells')

    return ad_sub


def get_merged_dataset(adata_all, obs):
    """
    :param adata_all: original adata with counts
    :param merged_obs: list of obs with leiden for each cell type
    :return: all_merged: merged adata with supercells
    """
    # counter number of super cells
    num_super_cells = 0
    
    num_super_cells += len(np.unique(obs.leiden))

    # merge datasets
    new_data = np.zeros((num_super_cells, adata_all.X.shape[1]))

    new_celltypes = []
    current_loc = 0
    sub_cell = adata_all[np.isin(adata_all.obs.index,
                                 obs.index)]
    num_current_cell = len(np.unique(obs.leiden))
    sub_cell.obs = obs
    merged = merge_cells(sub_cell)
    new_data[current_loc:(current_loc + num_current_cell), :] = merged.X
    current_loc += num_current_cell

    new_celltypes = np.array(new_celltypes)

    all_merged = sc.AnnData(new_data, obs=new_celltypes, var=adata_all.var)

    all_merged.X = np.round(all_merged.X)

    return all_merged

def supercell_pipeline(adata, ngenes=2000, npcs=20,ncell=500,verbose=True):
    saved_counts = adata.X.copy()
    
    # Run PCA and find nearest neighbors
    if verbose: print('preprocessing data...')
    sub_cells = preprocess_data(adata, n_genes=ngenes, npcs=npcs)
    
    if verbose: print('finding optimal resolution...')
    temp = get_leiden_based_on_ncell(sub_cells, ncell, verbose)

    adata.X = saved_counts
    
    sc.pp.normalize_total(adata, target_sum=1e4)
    if verbose: print('merging cells...')
    merged_data = get_merged_dataset(adata, temp.obs)
    
    return merged_data
