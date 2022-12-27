import pandas as pd
import numpy as np
from pyitlib import discrete_random_variable as drv
import networkx as nx
import os
from scipy.sparse import load_npz
from scipy.stats import chi2

from distributed import LocalCluster, Client
from dask import delayed
from dask.dataframe import from_delayed

class NetworkMerger:

    def __init__(self, adata, networks,
                 minimum_edge_appearance_threshold,prefix,
                 outdir, ncore, mem_per_core,
                 verbose):

        self.adata = adata
        self.dge = pd.DataFrame(adata.X.T)
        self.dge.index = self.adata.var.index
        
        self.networks = networks
        
        self.frac_networks = minimum_edge_appearance_threshold

        self.prefix = prefix
        self.outdir = outdir
        self.ncores = ncore
        self.memory_per_core = mem_per_core

        self.verbose = verbose


    def preprocess_network_files(self):
        """
        Takes the top 10 percent of networks, and top 3 parents of each node
        Finds edges in at least n% of networks.
        """
        
        self.print('Preprocessing in network files...')
        all_graphs = []

        top_percent = 0.1
        for temp in self.networks:
            # pandas has a bug with index col
            
            netsize = temp.shape[0]

            # add 3 potential edges per target, to reduce discrepencies due to feature importance
            df_top_genes = []
            for tup in temp.groupby('target'):
                # change this 3 if we want more parents per gene
                parents = np.min([len(tup[1]), 3])
                df_top_genes.append(tup[1].iloc[np.arange(parents)])
            df_top_genes = pd.concat(df_top_genes)

            temp = temp.iloc[np.arange(int(netsize*top_percent)),:]

            temp = pd.concat([temp, df_top_genes])

            temp = temp.drop_duplicates()            

            all_graphs.append(temp)

        # summarizing edges with weights
        self.print('Summarizing networks...')
        edge_weights = {}
        edge_number_appear = {}
        
        for g in all_graphs:
            for tup in g.itertuples():
                edge_index = (tup[2], tup[3])
                if edge_index not in edge_weights:
                    edge_weights[edge_index] = 0
                    edge_number_appear[edge_index] = 0
                edge_weights[edge_index] += tup[1]
                edge_number_appear[edge_index] += 1/len(all_graphs)

        summarized_network = pd.DataFrame(columns=['From', 'To', 'Weight', 'FractionAppeared'])
        for e in edge_weights:
            # only keep edges if in at least self.frac_networks fraction of the networks
            if edge_number_appear[e] > self.frac_networks:
                summarized_network.loc[summarized_network.shape[0], :] = [e[0], e[1],
                                                                          edge_weights[e],
                                                                          edge_number_appear[e]]

        self.summarized_network = summarized_network
        self.edge_weights = edge_weights

    def remove_reversed_edges(self):
        """
        Removes the reversed edge if the weight of the stronger edge is >25% over the weaker
        """
        self.print('Removing bidirectional edges...')
        indices_to_keep = []
        self.summarized_network.sort_values('Weight', inplace=True, ascending=False)
        counter = 0
        for tup in self.summarized_network.itertuples():
            reverse_edge = self.summarized_network[((self.summarized_network.loc[:, 'From'] == tup[2])
                                                    & (self.summarized_network.loc[:, 'To'] == tup[1]))]

            if reverse_edge.shape[0] > 0:
                # changed so we allow both directions
                weight_diff = tup[3] - reverse_edge.Weight.values.ravel()[0]
                if weight_diff > 0:
                    percent_diff = abs(weight_diff)/tup[3]
                else:
                    percent_diff = abs(weight_diff)/reverse_edge.Weight.values.ravel()[0]

                if percent_diff > 0.25:
                    if weight_diff > 0:
                        indices_to_keep.append(tup[0])
                    else:
                        indices_to_keep.append(reverse_edge.index[0])
                else:
                    indices_to_keep.append(tup[0])
                    indices_to_keep.append(reverse_edge.index[0])
            else:
                indices_to_keep.append(tup[0])
            counter += 1
        indices_to_keep = np.unique(np.array(indices_to_keep))
        self.summarized_network = self.summarized_network.loc[indices_to_keep, :]

    def remove_cycles(self):
        """
        Removes the weakest edge in each cycle
        """
        self.print('Removing cycles...')
        g = nx.DiGraph()
        for tup in self.summarized_network.itertuples():
            g.add_edge(tup[1], tup[2])

        edges_to_keep = []
        while True:
            try:
                cycle = nx.algorithms.cycles.find_cycle(g, orientation='original')
            except:
                break

            edge_removal = cycle[0]
            edge_removal_score = 1e9
                
            for i, edge in enumerate(cycle):
                edge_score = self.edge_weights[(edge[0], edge[1])]
                if edge_score < edge_removal_score:
                    edge_removal = edge
                    edge_removal_score = edge_score

            if len(cycle) == 2:
                edges_to_keep.append((edge_removal[0], edge_removal[1]))

            g.remove_edge(edge_removal[0], edge_removal[1])

        weight_ = []
        from_ = []
        to_ = []
        for edge in g.edges:
            weight = self.summarized_network[(self.summarized_network['From'] == edge[0]) & (self.summarized_network['To'] == edge[1])]['Weight'].values.ravel()[0]
            weight_.append(weight)
            from_.append(edge[0])
            to_.append(edge[1])

        for edge in edges_to_keep:
            weight = self.summarized_network[(self.summarized_network['From'] == edge[0]) & (self.summarized_network['To'] == edge[1])]['Weight'].values.ravel()[0]
            weight_.append(weight)
            from_.append(edge[0])
            to_.append(edge[1])

        self.edge_df = pd.DataFrame({"source": from_, "target": to_, "importance" : weight_})

    def get_triads(self):
        """
        Finds triads in which 3 genes have a triangle structure
        """
        self.print('Getting triads to remove redundant edges...')
        source_genes = np.unique(self.edge_df.source)

        # get subtrees
        subtrees = {}
        for g1 in source_genes:
            subtree = self.edge_df[self.edge_df.source == g1]
            if subtree.shape[0] >= 2:
                subtrees[g1] = subtree

        # find triads
        triads_to_evaluate = []
        for g1 in subtrees:
            subtree = subtrees[g1]

            # nested iteration through
            for g2 in subtree.target:
                for g3 in subtree.target:
                    if g2 != g3:
                        # make sure g2 is a source gene
                        if g2 in subtrees:
                            if g3 in subtrees[g2].target.values.ravel():
                                triads_to_evaluate.append(
                                    [(g1, g2), (g1, g3), (g2, g3)])

        self.triads_to_evaluate = triads_to_evaluate

    def remove_redundant_edges(self):
        """
        Removes edges in triads if the conditional mutual information is 
        conditionally independent with a third node
        """
        computed_edges_to_remove = None
        self.print('Removing redundant edges...')
        try:
            loc_cluster = LocalCluster(n_workers=self.ncores,
                                       threads_per_worker=1,
                                       memory_limit=self.memory_per_core)

            self.print('Creating client...')
            client = Client(loc_cluster)
            client, shutdown_callback = _prepare_client(client)

            self.print('Loading data...')
            delayed_matrix = client.scatter(self.dge, broadcast=True)
            alpha = 0.05 # significance threshold
            edges_to_remove = []

            self.print('Building dask graph...')
            for t in self.triads_to_evaluate:
                delayed_edge = delayed(mi_computation, pure=True)(
                    delayed_matrix, t, alpha
                )

                edges_to_remove.append(delayed_edge)

            n_parts = len(client.ncores())
            edges_removal_df = from_delayed(edges_to_remove)
            all_removal_edges = edges_removal_df.repartition(npartitions=n_parts)
            self.print('Computing dask graph...')
            computed_edges_to_remove = client.compute(all_removal_edges, sync=True)

        except:
            self.print('Closing client...')
            client.close()
            loc_cluster.close()

        self.print('Removing edges...')
        if computed_edges_to_remove is not None:
            if 'blank' in computed_edges_to_remove.source.to_numpy().ravel():
                computed_edges_to_remove = computed_edges_to_remove[computed_edges_to_remove.source != 'blank']

            inds_to_remove = []
            for e in computed_edges_to_remove.itertuples():
                inds_to_remove.append(
                    np.intersect1d(np.where(self.edge_df.source == e[1]),
                                   np.where(self.edge_df.target == e[2]))[0])
            inds_to_remove = np.unique(inds_to_remove)

            locs_to_keep = np.where(np.isin(np.arange(self.edge_df.shape[0]), inds_to_remove, invert=True))[0]

            self.edge_df = self.edge_df.iloc[locs_to_keep, :]

    def save_network(self):
        if self.outdir[-1] != '/':
            self.outdir = self.outdir + '/'

        self.print('Saving data to '+self.outdir)
        os.makedirs(self.outdir, exist_ok=True)
        self.edge_df.to_csv(self.outdir+self.prefix+'.network.merged.csv',
                            index=False)

    def print(self, input_str):
        if self.verbose:
            print(input_str)

    def pipeline(self):

        self.preprocess_network_files()
        
        self.remove_reversed_edges()

        self.remove_cycles()

        self.get_triads()
        self.remove_redundant_edges()

        self.save_network()


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

def quantile_variables(x, nquantiles):
    quant_data = np.zeros_like(x)
    x_without_zero = list(x[x!=0])
    x_without_zero.append(0)
    x_without_zero = np.array(x_without_zero)
    quantiles = np.quantile(x_without_zero, np.linspace(0,1,nquantiles + 1))
    
    for q in range(1,len(quantiles)):
        if q == 1:
            quant_data[x <= quantiles[q]] = 0
        else:
            quant_data[((x > quantiles[q-1])&
                       (x <= quantiles[q]))] = int(q-1)

    return quant_data


def get_mi_p_chisquare(a, b, c):
    """
        From https://jmlr.csail.mit.edu/papers/volume22/19-600/19-600.pdf
        Kubkowski et al.
        2n*I(X,Y|Z) > chi_squared(d)
        where d = (I-1)(J-1)(K)
        I, J, K are the possible values in X, Y, and Z respectively
    """
    nquants = 3
    a = quantile_variables(a, nquants)
    b = quantile_variables(b, nquants)
    c = quantile_variables(c, nquants)
    
    # d = (I-1)(J-1)K
    d = (nquants - 1)*(nquants - 1)*nquants

    conditional_mi = drv.information_mutual_conditional(b, c, a)
    value_to_test_chi_squared = len(a) * 2 * conditional_mi
    
    chi_cdf = chi2.cdf(value_to_test_chi_squared, d)
    
    conditional_mi_pval = 1 - chi_cdf

    return conditional_mi_pval


def mi_computation(dge_input, t, alpha):
    x = dge_input.loc[t[0][0], :].values.ravel()
    y = dge_input.loc[t[0][1], :].values.ravel()
    z = dge_input.loc[t[1][1], :].values.ravel()

    conditional_mi_pval_y_z_given_x = get_mi_p_chisquare(x.copy(), y.copy(), z.copy())
    if conditional_mi_pval_y_z_given_x > alpha:
        edge_to_remove = [t[2][0], t[2][1]]
    else:
        conditional_mi_pval_x_z_given_b = get_mi_p_chisquare(x.copy(), z.copy(), y.copy())
        if conditional_mi_pval_x_z_given_b > alpha:
            edge_to_remove = [t[1][0], t[1][1]]
        else:
            edge_to_remove = ['blank', 'blank']
    edge_to_remove = pd.DataFrame(edge_to_remove).T
    edge_to_remove.columns = ['source', 'target']
    return pd.DataFrame(edge_to_remove)
