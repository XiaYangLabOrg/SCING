from supercell_helpers import *
import argparse
import scanpy as sc
import pandas as pd
import numpy as np
import os

# read in command line arguments
parser = argparse.ArgumentParser(description='Merges cells into supercells',
                                 add_help=True)

# Digital gene expression matrix
parser.add_argument('expression', metavar='data_file',
                    type=str, nargs=1, action='store',
                    help='Gene Expression Matrix csv. Genes x Cells')
# celltypes
parser.add_argument('celltypes', metavar='celltype_file',
                    type=str, nargs=1, action='store',
                    help='Celltype vector file. Each row is a cell type')

# ngenes
parser.add_argument('--ngenes',
                    type=int,
                    action='store',
                    help='Number of genes to use when finding cell clusters (default: 2000)',
                    default=2000)

# npcs
parser.add_argument('--pcs',
                    type=int,
                    action='store',
                    help='Number of PCs to use when finding cell clusters (default: 40)',
                    default=40)

# thresh
parser.add_argument('--threshold',
                    type=float,
                    action='store',
                    help='Either ncell, percent zeros, or variance threshold to merge cells to. Default is for ncell '
                         '(default: 25)',
                    default=25)

# percent cells per dataset
parser.add_argument('--percent_cells',
                    type=float,
                    action='store',
                    help='Percent cells to use per dataset for MCMC (default: 0.7)',
                    default=0.7)

# iteration of method
parser.add_argument('--iteration',
                    type=int,
                    action='store',
                    help='iteration of method to MCMC sampling, if not applicable leave at 0 (default: 0)',
                    default=0)

# ncell, percent or var
parser.add_argument('--type',
                    type=str,
                    action='store',
                    help='ncell, var, or percent: decides which metric to use for merging (default: ncell)',
                    default='ncell')

# verbose
parser.add_argument('--verbose',
                    type=str,
                    action='store',
                    help='Print output or not (default: False)',
                    default=False)

# outdir
parser.add_argument('--outdir',
                    type=str,
                    action='store',
                    help='directory to output supercell files',
                    default='processed_supercells')

args = parser.parse_args()

cellt_file = args.celltypes[0]
gene_file = args.expression[0]
ngene = args.ngenes
npcs = args.pcs
threshold = args.threshold
method_type = args.type
percent_cells = args.percent_cells
iteration = args.iteration
verbose = args.verbose == 'True'
outdir = args.outdir

# read in data
if verbose:
    print('Reading in data')
data = pd.read_csv(gene_file, sep=',', index_col=0)
celltypes = pd.read_csv(cellt_file, header=None).to_numpy().ravel()
cells = np.unique(celltypes)

adata = sc.AnnData(data.to_numpy().T,
                   obs=celltypes,
                   var=data.index.to_numpy().ravel())

adata.obs.columns = ['celltypes']
adata.var.columns = ['genes']

saved_counts = adata.X.copy()

# preprocess data
if verbose:
    print('Preprocessing data')

cell_adatas = []
for c in cells:
    if verbose:
        print(c)
    sub_cells = adata[adata.obs.celltypes == c]
    sub_cells = preprocess_data(sub_cells, n_genes=ngene, npcs=npcs, percent_cells=percent_cells)
    cell_adatas.append(sub_cells)

# merge cells
if verbose:
    print('Finding resolution to merge each cell type')

resolutions = np.arange(0.1, 1000, 0.1)
merged_cell_obs = []
for i, c in enumerate(cells):
    if verbose:
        print(c)
    if method_type == 'ncell':
        temp = get_leiden_based_on_ncell(cell_adatas[i], resolutions, threshold, verbose)
    elif method_type == 'var':
        temp = merge_cells_and_check_percent_var(cell_adatas[i], resolutions, threshold, verbose)
    else:
        temp = merge_cells_and_check_percent_zeros(cell_adatas[i], resolutions, threshold, verbose)
    merged_cell_obs.append(temp.obs)

adata.X = saved_counts
sc.pp.normalize_total(adata, target_sum=1e4)
merged_data = get_merged_dataset(adata, merged_cell_obs)

if not os.path.exists(outdir):
    if verbose:
        print(outdir + ' does not exist...')
        print('creating ' + outdir)
    os.makedirs(outdir)

if verbose:
    print('saving files to ' + outdir)

pd.DataFrame(merged_data.X).to_csv(outdir + '/supercells.counts.' + str(threshold) + '.' + str(iteration) + '.csv.gz',
                                   index=False, header=False, compression='gzip')
merged_data.obs.celltypes.to_csv(outdir + '/supercells.celltypes.' + str(threshold) + '.' + str(iteration) + '.csv.gz',
                                 index=False, header=False, compression='gzip')
merged_data.var.genes.to_csv(outdir + '/supercells.genes.' + str(threshold) + '.' + str(iteration) + '.csv.gz',
                             index=False, header=False, compression='gzip')
