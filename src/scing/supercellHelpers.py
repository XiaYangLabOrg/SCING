import scanpy as sc
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from collections import Counter


def preprocess(adata, n_hvgs=2000, n_pcs=40, n_neighbors=10):
    """
    Preprocess scRNAseq data (log-normalize, hvg, scale, pca, KNN)

    Args:
        adata: AnnData scRNAseq
        n_hvgs: number of highly variable genes
        n_pcs: number of PCs
        n_neighbors: number of neighbors for KNN
    
    Returns:
        ad_sub: AnnData object of processed scRNAseq data with log-normalized counts
    """

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=n_hvgs)
    adata = adata[:, adata.var.highly_variable]
    ad_sub = sc.AnnData(adata.X, obs=adata.obs, var=adata.var)
    saved_log_mat = ad_sub.X.copy()
    sc.pp.scale(ad_sub)
    sc.tl.pca(ad_sub, svd_solver='arpack', n_comps=min(n_pcs, adata.shape[0] - 1))
    sc.pp.neighbors(ad_sub, n_neighbors=n_neighbors, n_pcs=min(n_pcs, adata.shape[0] - 1))
    ad_sub.X = saved_log_mat
    return(ad_sub)

def get_knn_adj_list(adata, n_neighbors=10):
    """
    Get cell KNN Adjacency List from adata object. 
    
    Args:
        adata: AnnData scRNAseq object with KNN connecivites stored in adata.obsp['connectivities']
        n_neighbors: number of neighbors to use in pseudobulking. Cell is included as its own neighbor.
    
    Returns:
        K: KNN adjacency list (cells x n_neighbor)
    """

    K = np.argsort(adata.obsp['connectivities'].toarray(), axis=1)[:,::-1]
    K = np.concatenate([np.arange(K.shape[0]).reshape(K.shape[0],1),K[:,:(n_neighbors-1)]], axis=1)
    return K

def find_pb_cells(K, n_neighbors=10, max_overlap=5, random_state=0):
    """
    Find cell neighbors for pseudobulking

    Args:
        K: KNN adjacency list
        n_neighbors: number of neighbors to pseudobulk (must be <= #nearest neighbors in KNN preprocessing step)
        max_overlap: max cell overlap between 2 pseudobulk cells (must be < n_neighbors)
    
    Returns:
        P: list of cell indices to perform pseudobulking on
    """
    # cells to sample from
    cells = list(np.arange(K.shape[0]))
    np.random.seed(random_state)
    P=[] # list of pb cell indices
    # find pb cells until max number is reached, or no more cells remain
    while len(cells)>0:
        if len(P)%200 == 0:
            print(f"{len(P)} pseudobulk cells made... {len(cells)} cells remaining")
        c = np.random.choice(cells)
        if len(P)>0 and np.max(np.isin(K[P,:n_neighbors],K[c,:n_neighbors]).sum(axis=1)) >= max_overlap:
            # print(f"skipping cell {c}")
            cells.remove(c)
            continue
        P.append(c)
        cells.remove(c)
        
    print(f"There are {len(P)} pseudobulk cells")
    return P

def aggregate_cells(adata_raw, cells, method="average"):
    """
    Aggregates cells from adata object

    Args:
        adata_raw: AnnData scRNAseq object with raw counts
        cells: list of cell names
        method: count aggregation method (must be 'average' or 'sum')
    
    Returns:
        1D array of pseudobulk counts

    """
    if method=='average':
        return adata_raw[cells,].X.mean(axis=0)
    elif method=='sum':
        return adata_raw[cells,].X.sum(axis=0)
    else:
        raise ValueError("method must be sum or average")

def smooth_pseudobulk(adata_raw, P, K, celltype_names, n_pb=500, random_state=0):
    """
    Generate pseudobulk matrix using aggregated expression of cells and K-nearest neighbors.

    Args:
        adata_raw: AnnData scRNAseq object with raw counts
        P: list of cell indices to perform cell pseudobulking
        K: KNN adjacency list for each cell
        celltype_names: pd.Index or array of cell names
        n_pb: number of pseudobulk cells to generate
        random_state: random state for setting seed
    
    Returns:
        adata_pb: AnnData scRNAseq object of pseudobulk cells
    """
    
    np.random.seed(random_state)
    n_pb = min(n_pb,len(P)) # number of pseudobulk cells
    
    # initialize new counts matrix
    print(f"Generating {n_pb} pseudobulk cells")
    new_mat = np.empty((0,adata_raw.shape[1]))
    # randomly select pseudobulk cells to generate, concatenate to new matrix
    for count, i in enumerate(np.random.choice(P, n_pb, replace=False).tolist()):
        neighbors = K[i,:]
        new_mat = np.concatenate([new_mat,aggregate_cells(adata_raw, celltype_names[neighbors])],
                                  axis=0)
        if count%100==0:
            print(f"{count}/{n_pb} done") 
    # AnnData pseudobulk object
    adata_pb = sc.AnnData(X=csr_matrix(new_mat), var=pd.DataFrame(index=adata_raw.var_names), dtype=np.float32)
    # gene sparsity
    adata_pb.var['sparsity'] = np.sum(adata_pb.X.toarray()==0, axis=0)/adata_pb.shape[0]
    adata_pb = adata_pb[:,adata_pb.var['sparsity']<1]
    print(f"removed {adata_raw.shape[1] - adata_pb.shape[1]} genes with no expression")
    
    return(adata_pb)

def pseudobulk_pipeline(adata, ):
    return



# Deprecated functions
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
