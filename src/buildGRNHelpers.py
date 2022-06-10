import scanpy as sc
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
from distributed import LocalCluster, Client
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from dask import delayed
from dask.dataframe import from_delayed

# class taken from GRNBOOST2
class EarlyStopMonitor:
    def __init__(self, window_length=25):
        """
        :param window_length: length of the window over the out-of-bag errors.
        """

        self.window_length = window_length

    def window_boundaries(self, current_round):
        """
        :param current_round:
        :return: the low and high boundaries of the estimators window to consider.
        """

        lo = max(0, current_round - self.window_length + 1)
        hi = current_round + 1

        return lo, hi

    def __call__(self, current_round, regressor, _):
        """
        Implementation of the GradientBoostingRegressor monitor function API.
        :param current_round: the current boosting round.
        :param regressor: the regressor.
        :param _: ignored.
        :return: True if the regressor should stop early, else False.
        """

        if current_round >= self.window_length - 1:
            lo, hi = self.window_boundaries(current_round)
            return np.mean(regressor.oob_improvement_[lo: hi]) < 0
        else:
            return False


# function taken from arboreto/GRNBOOST2
def _prepare_client(client_or_address):
    """
    :param client_or_address: one of:
           * None
           * verbatim: 'local'
           * string address
           * a Client instance
    :return: a tuple: (Client instance, shutdown callback function).
    :raises: ValueError if no valid client input was provided.
    """

    if client_or_address is None or str(client_or_address).lower() == 'local':
        local_cluster = LocalCluster(diagnostics_port=None)
        client = Client(local_cluster)

        def close_client_and_local_cluster(verbose=False):
            if verbose:
                print('shutting down client and local cluster')

            client.close()
            local_cluster.close()

        return client, close_client_and_local_cluster

    elif isinstance(client_or_address, str) and client_or_address.lower() != 'local':
        client = Client(client_or_address)

        def close_client(verbose=False):
            if verbose:
                print('shutting down client')

            client.close()

        return client, close_client

    elif isinstance(client_or_address, Client):

        def close_dummy(verbose=False):
            if verbose:
                print('not shutting down client, client was created externally')

            return None

        return client_or_address, close_dummy

    else:
        raise ValueError("Invalid client specified {}".format(str(client_or_address)))


class grnBuilder:
    def __init__(self, adata, ngenes, nneighbors, npcs,
                 prefix, outdir, ncore, mem_per_core, verbose):

        self.adata = adata
        self.dge = pd.DataFrame(adata.X.T)
        self.dge.index = self.adata.var.index
        
        self.ngenes = ngenes
        self.nneighbors = nneighbors
        self.npcs = npcs

        self.prefix = prefix
        if outdir[-1] != '/':
            outdir = outdir + '/'
            
        self.outdir = outdir

        self.ncores = ncore
        self.memory = mem_per_core

        self.verbose = verbose
        file_name = self.outdir + self.prefix + '.csv.gz'

            
    def pipeline(self):

        self.print('Filtering genes...')
        self.filter_genes()

        if self.nsupercells is not None:
            self.merge_cells()

        self.print('Filtering gene connectivities...')
        self.filter_gene_connectivities()
        self.print('Building GRN...')
        self.build_grn()
        self.print('Saving data...')
        self.save_edges()

    def print(self, string_to_print):
        if self.verbose:
            print(string_to_print)

    def filter_genes(self):
        adata = sc.AnnData(self.dge.copy()).T
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        print(adata)
        print((~dge.duplicated()).to_numpy().ravel())
        print((~dge.duplicated()).to_numpy().ravel().shape)
        adata = adata[:,(~dge.duplicated()).to_numpy().ravel()]


        adata = adata[:,np.sum(adata.X,axis=0) != 0]       

        # if ngenes is -1 or more than the total number of genes, no filtering is done
        if ((self.ngenes == -1) | (self.ngenes >= self.dge.shape[0])):
            self.adata = adata
            return

        sc.pp.highly_variable_genes(adata, n_top_genes=self.ngenes)
        genes_to_keep = adata.var.index[np.where(adata.var.highly_variable)]

        self.dge = self.dge.loc[genes_to_keep, :]

        adata = adata[:, adata.var.highly_variable]
        self.adata = adata

    def merge_cells(self):
        adata = get_leiden_based_on_ncell(self.adata,
                                          resolutions=np.arange(0.001, 1000, 0.1),
                                          num_cells=self.nsupercells,
                                          verbose=self.verbose)

        new_adata = sc.AnnData(self.dge).T
        new_adata.obs['leiden'] = adata.obs.leiden
        
        # using a second merged dataset
        # this is for the average counts
        # the other is based on normalized results
        new_merged_adata = merge_cells(new_adata)
        new_dge = pd.DataFrame(new_merged_adata.X).T

        new_dge.index = new_merged_adata.var.index
        new_dge.columns = new_merged_adata.obs.index
        self.dge = new_dge
        
        self.adata = merge_cells(adata)

    def filter_gene_connectivities(self):
        self.adata = self.adata.copy().T
        #mat = (self.adata.X > 0).astype(np.float64)

        sc.pp.scale(self.adata, max_value=10)
        sc.tl.pca(self.adata, svd_solver='arpack', n_comps=self.npcs)
        #sc.pp.neighbors(self.adata, n_neighbors=self.nneighbors, n_pcs=100, method='gauss')
        
        clf = NearestNeighbors(n_neighbors=self.nneighbors+1,
                              n_jobs=self.ncores).fit(self.adata.obsm['X_pca'])
        dist, neighbor_inds = clf.kneighbors(self.adata.obsm['X_pca'])
        dist = dist[:,1:]
        neighbor_inds = neighbor_inds[:, 1:]
        
        gene_connectivities = np.zeros((self.adata.obs.shape[0],
                                       self.adata.obs.shape[0]))
        
        gene_connectivities = pd.DataFrame(gene_connectivities)
        gene_connectivities.index = self.adata.obs.index
        gene_connectivities.columns = self.adata.obs.index
        
        for i in range(neighbor_inds.shape[0]):
            dist_vec = dist[i,:]
            ind_vec = neighbor_inds[i,:]
            #ind_vec = ind_vec[dist_vec < self.connect_thresh]

            gene_connectivities.iloc[i, ind_vec] = 1
        
        self.gene_connectivities = gene_connectivities

        del self.adata

        if self.tftg_file is not None:
            tftg = pd.read_csv(self.tftg_file)

            # gets TFTG where both genes are there
            tftg = tftg.loc[np.sum(np.isin(tftg,self.gene_connectivities.index),axis=1) == 2,:]

            for tup in tftg.itertuples():
                self.gene_connectivities.loc[tup[1],tup[2]] = 1
                self.gene_connectivities.loc[tup[2],tup[1]] = 0 

    def build_grn(self):
        if self.grn_type == 'intra':
            genes_to_compute = self.gene_connectivities.index.to_numpy().ravel()
            sender_genes = genes_to_compute
            receiver_genes = genes_to_compute
        elif self.grn_type == 'inter':
            genes_to_compute = self.gene_connectivities.index.to_numpy().ravel()
            all_celltypes_in_genes = np.array([i.split("_")[0] for i in genes_to_compute])
            celltypes = np.unique(all_celltypes_in_genes)
            sender_cell = celltypes[0]
            sender_genes = genes_to_compute[[sender_cell in i for i in all_celltypes_in_genes]]
            receiver_cell = celltypes[1]
            receiver_genes = genes_to_compute[[receiver_cell in i for i in all_celltypes_in_genes]]

        else:
            print('grn_type needs to either be intra or inter')
            quit()

        self.sender_genes = sender_genes
        self.receiver_genes = receiver_genes

        self._build_grn()

    def _build_grn(self):
        try:
            threads_per_core = 1

            self.print('building local cluster')
            loc_cluster = LocalCluster(n_workers=self.ncores,
                                       threads_per_worker=threads_per_core,
                                       memory_limit=self.memory)

            client = Client(loc_cluster)

            client, shutdown_callback = _prepare_client(client)

            self.print('Loading data into memory...')

            delayed_matrix = client.scatter(self.dge, broadcast=True)

            delayed_link_df = []

            early_stop_window_length = 25
            counter = 0

            self.print('Building dask graph...')
            for g1 in self.receiver_genes:
                connectivity_vec = self.gene_connectivities.loc[g1, :]
                connectivity_vec = connectivity_vec[self.sender_genes]
                connectivity_vec = connectivity_vec[connectivity_vec > 0]

                if len(connectivity_vec) > 0:
                    delayed_reg = delayed(grad_boost_reg, pure=True)(delayed_matrix, g1,
                                                                     connectivity_vec.index,
                                                                     early_stop_window_length)

                    delayed_link_df.append(delayed_reg)

                counter += 1
            n_parts = len(client.ncores()) * threads_per_core
            edges_df = from_delayed(delayed_link_df)
            all_links_df = edges_df.repartition(npartitions=n_parts)

            self.print('Computing dask graph...')

            computed_edges = client.compute(all_links_df, sync=True).sort_values(by='importance', ascending=False)

            self.edges = computed_edges

        finally:
            self.print('Closing client...')
            client.close()
            loc_cluster.close()

    def save_edges(self):
        os.makedirs(self.outdir, exist_ok=True)
        file_name = self.outdir + self.prefix + '.' + self.grn_type + '.csv.gz'
        self.print('Saving file in '+file_name)
        self.edges.to_csv(file_name)



def grad_boost_reg(temp_dge_input, target_gene, input_genes, early_stop_window_length):
    gene_vec_input = temp_dge_input.loc[target_gene, :]

    
    reg_rf = GradientBoostingRegressor(n_estimators=500,
                                       max_depth=3,
                                       learning_rate=0.01,
                                       subsample=0.9,
                                       max_features='sqrt',
                                       verbose=0).fit(temp_dge_input.loc[input_genes, :].T,
                                                      gene_vec_input,
                                                      monitor=EarlyStopMonitor(early_stop_window_length))

    reg_rf = pd.DataFrame([[i[0], i[1]] for i in sorted(zip(reg_rf.feature_importances_, input_genes))][::-1])
    reg_rf.columns = ['importance', 'source']
    reg_rf.loc[:, 'target'] = target_gene
    reg_rf = reg_rf[reg_rf.importance != 0]
    reg_rf.importance = reg_rf.importance * len(input_genes)

    return reg_rf


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


def get_leiden_based_on_ncell(ad_sub, resolutions, num_cells, verbose):
    """
    :param ad_sub: adata of single celltype
    :param resolutions: np vector of resolutions to test
    :param num_cells: average number of cells per cluster
    :param verbose: Whether or not to print along the way
    :return: ad_sub: adata of single celltype with clusters in leiden
    """
    saved_log_mat = ad_sub.X.copy()
    sc.pp.scale(ad_sub)
    sc.tl.pca(ad_sub, svd_solver='arpack', n_comps=min(30, ad_sub.shape[0] - 1))
    sc.pp.neighbors(ad_sub, n_neighbors=10, n_pcs=min(30, ad_sub.shape[0] - 1))
    ad_sub.X = saved_log_mat

    vec_length = len(resolutions)
    iter_ = int(vec_length / 2)
    last_iter = 0
    while True:
        vec_length = abs(iter_ - last_iter)
        sc.tl.leiden(ad_sub, resolution=resolutions[iter_], n_iterations=100)

        # get number of cells per group on average
        num_groups = len(np.unique(ad_sub.obs.leiden))


        if abs(iter_ - last_iter) <= 1:
            break

        last_iter = iter_
        if num_groups <= num_cells:
            iter_ += int(vec_length / 2)
        elif num_groups > num_cells:
            iter_ -= int(vec_length / 2)

    sc.tl.leiden(ad_sub, resolution=resolutions[iter_])

    if verbose:
        print('There are ',
              len(np.unique(ad_sub.obs.leiden.to_numpy().ravel())),
              ' subgroups')

    return ad_sub
