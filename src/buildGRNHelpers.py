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


class grnBuilder:
    def __init__(self, adata, ngenes, nneighbors, npcs, subsample_perc,
                 prefix, outdir, ncore, mem_per_core, verbose):

        self.adata = adata
        self.dge = pd.DataFrame(adata.X.T)
        self.dge.index = self.adata.var.index
        
        self.ngenes = ngenes
        self.nneighbors = nneighbors
        self.npcs = npcs
        self.subsample_percentage = subsample_perc

        self.prefix = prefix
        if outdir[-1] != '/':
            outdir = outdir + '/'
            
        self.outdir = outdir

        self.ncores = ncore
        self.memory = mem_per_core * ncore

        self.verbose = verbose
        file_name = self.outdir + self.prefix + '.csv.gz'

            
    def pipeline(self):
        self.subsample_cells()
        
        self.print('Filtering genes...')
        self.filter_genes()

        self.print('Filtering gene connectivities...')
        self.filter_gene_connectivities()
        self.print('Building GRN...')
        self.build_grn()
        self.print('Saving data...')
        self.save_edges()

    def print(self, string_to_print):
        if self.verbose:
            print(string_to_print)
                  
    def subsample_cells(self):
        if self.subsample_percentage >= 1:
            return
        if self.subsample_percentage <= 0:
            print('Please put a subsample percentage between 0 and 1')
            quit()

        self.dge = self.dge.sample(frac=self.subsample_percentage,
                                   axis=1,
                                   replace=False)


    def filter_genes(self):
        adata = sc.AnnData(self.dge.copy()).T
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
        # remove duplicates
        adata = adata[:,(~self.dge.duplicated()).to_numpy().ravel()]

        # remove genes without expression
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


    def filter_gene_connectivities(self):
        # transpose to operate on genes
        self.adata = self.adata.copy().T

        sc.pp.scale(self.adata, max_value=10)
        sc.tl.pca(self.adata, svd_solver='arpack', n_comps=self.npcs)
        
        # find nearest neighbors of each gene in PC space
        clf = NearestNeighbors(n_neighbors=self.nneighbors+1,
                              n_jobs=self.ncores).fit(self.adata.obsm['X_pca'])
        dist, neighbor_inds = clf.kneighbors(self.adata.obsm['X_pca'])
        dist = dist[:,1:]
        neighbor_inds = neighbor_inds[:, 1:]
        
        # limits connctivities between genes
        gene_connectivities = np.zeros((self.adata.obs.shape[0],
                                       self.adata.obs.shape[0]))
        
        gene_connectivities = pd.DataFrame(gene_connectivities)
        gene_connectivities.index = self.adata.obs.index
        gene_connectivities.columns = self.adata.obs.index
        
        for i in range(neighbor_inds.shape[0]):
            dist_vec = dist[i,:]
            ind_vec = neighbor_inds[i,:]

            # add neighbors of each gene to possible connectivity
            gene_connectivities.iloc[i, ind_vec] = 1
        
        self.gene_connectivities = gene_connectivities

        del self.adata


    def build_grn(self):
        genes_to_compute = self.gene_connectivities.index.to_numpy().ravel()
        sender_genes = genes_to_compute
        receiver_genes = genes_to_compute


        self.sender_genes = sender_genes
        self.receiver_genes = receiver_genes

        self._build_grn()

    def _build_grn(self):
        # early stopping for GBR
        early_stop_window_length = 25
        if self.ncores > 1:
            try:
                threads_per_core = 1


                self.print('building local cluster')
                # dask set up
                loc_cluster = LocalCluster(n_workers=self.ncores,
                                           threads_per_worker=threads_per_core,
                                           memory_limit=self.memory)

                client = Client(loc_cluster)

                client, shutdown_callback = _prepare_client(client)

                self.print('Loading data into memory...')

                delayed_matrix = client.scatter(self.dge, broadcast=True)

                delayed_link_df = []

                
                counter = 0

                self.print('Building dask graph...')
                for g1 in self.receiver_genes:
                    # genes to add as potential parents are limited by the connectivity matrix
                    connectivity_vec = self.gene_connectivities.loc[g1, :]
                    connectivity_vec = connectivity_vec[self.sender_genes]
                    connectivity_vec = connectivity_vec[connectivity_vec > 0]

                    if len(connectivity_vec) > 0:
                        # gradient boosting regressor
                        delayed_reg = delayed(grad_boost_reg, pure=True)(delayed_matrix, g1,
                                                                         connectivity_vec.index,
                                                                         early_stop_window_length)

                        delayed_link_df.append(delayed_reg)

                    counter += 1
                n_parts = len(client.ncores()) * threads_per_core
                # merge the edges together
                edges_df = from_delayed(delayed_link_df)
                all_links_df = edges_df.repartition(npartitions=n_parts)

                self.print('Computing dask graph...')

                computed_edges = client.compute(all_links_df, sync=True).sort_values(by='importance', ascending=False)

                self.edges = computed_edges

            finally:
                self.print('Closing client...')
                client.close()
                loc_cluster.close()
        else:
            edges_df = []
            for g1 in self.receiver_genes:
                connectivity_vec = self.gene_connectivities.loc[g1, :]
                connectivity_vec = connectivity_vec[self.sender_genes]
                connectivity_vec = connectivity_vec[connectivity_vec > 0]
                if len(connectivity_vec) > 0:
                    # gradient boosting regressor
                    delayed_reg = grad_boost_reg(self.dge, g1,
                                                 connectivity_vec.index,
                                                 early_stop_window_length)

                    edges_df.append(delayed_reg)
            self.edges = pd.concat(edges_df).sort_values(by='importance', ascending=False)
                

    def save_edges(self):
        os.makedirs(self.outdir, exist_ok=True)
        file_name = self.outdir + self.prefix + '.csv.gz'
        self.print('Saving file in '+file_name)
        self.edges.to_csv(file_name)

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

    reg_rf_df = pd.DataFrame([[i[0], i[1]] for i in sorted(zip(reg_rf.feature_importances_, input_genes))][::-1])
    #print(reg_rf_df)
    reg_rf_df.columns = ['importance', 'source']
    reg_rf_df.loc[:, 'target'] = target_gene
    reg_rf_df = reg_rf_df[reg_rf_df.importance != 0].copy()
    reg_rf_df.loc[:,'importance'] = reg_rf_df.loc[:,'importance'].to_numpy().ravel()*len(input_genes)

    return reg_rf_df
