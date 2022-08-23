# SCING
SCING: Single Cell INtegrative Gene regulatory network inference elucidates robust, interpretable gene regulatory networks

![alt text](https://github.com/XiaYangLabOrg/SCING/blob/main/img/SCING.overview.figure.png)

SCING overview, benchmarking, and application. SCING overview (a). First, we select a specific cell type, or use whole visium data. We then cluster the cells using the leiden graph partitioning algorithm, and merge subclusters into supercells. We utilize bagging through subsamples of supercells to keep robust edges in the final GRN. For each subsample, the genes are clustered to only operate on likely regulatory edges. We then identify edges through gradient boosting regressors (GBR). We find the consensus as edges that show up in 20% of the networks. We then prune edges and cycles using conditional mutual information petrics. Perturb-seq validation (b). We identified downstream perturbed genes of guides for specific genes. We then predict perturbed genes at each depth in the network from the perturbed gene. True positive rate, and false positive rate are determined at each depth in the network. We utilize AUROC and TPR at FPR 0.05 as metrics for evaluation. Gene prediction validation, to determine network overfitting (b). We split data into train and test sets, and built a network on the train set. A GBR is trained for each gene based on its parents in the train data. We then predict the expression of each gene in the test set and determine the distance from the true expression through cosine similarity. Biological validation through disease subnetwork modeling (c). We utilize a random walk framework from Huang et al. to determine the increase in performance of a GRN to model disease subnetworks versus a random GRN with similar node attributes (d). We utilize the leiden graph partitioning algorithm to identify GRN subnetworks. We combine these subnetworks with the AUCell method to get module specific expression for each cell, and further combine the gene modules with pathway knowledge bases to annotate modules with biological pathways. We apply SCING to human prefrontal cortex snRNAseq data with AD and Control patients, whole brain visium data, for AD vs WT mice at different ages, and to the mouse cell atlas in 33 tissues and 106 cell types (f). 

  
  ---
  ### Downloading the repo  
  ``` 
  $git clone https://github.com/XiaYangLabOrg/SCING.git
  $cd SCING
  ```  
  
  ### Setting up the environment:
  ``` 
  $conda env create -n scing --file install/scing.environment.yml  
  $conda activate scing
  $pip install pyitlib  
  ```
  
  If you want to use the AUCell from SCENIC for graph based dimensionality reduction you must install pyscenic  
  ```
  $pip install pyscenic
  ```
  
  ### Running SCING  
  #### Tutorials can be found in the tutorials directory  
  BuildNetwork.ipynb: Build Supercells, build many GRNs based on subsamples of the data, merge many GRNs to a final GRN plus prune edges.  
  *Note: To properly use the parallelizability of SCING with a cluster. These steps can be broken up into three separate scripts. First, build supercells and save the file. Second, for 100 networks (in parallel) read in the supercell and build GRNs. Lastly, read in the supercells, and the 100 networks, and merge the networks.*  
    
  ModuleBasedDimensionalityReduction.ipynb: Take the merged network, and divide it into subnetworks with the leiden graph partitioning algorithm, finally run the AUCell approach to get module scores for each cell in the dataset. These can be used for clustering or phenotypic association.  
    
  PathwayEnrichmentOfModule.ipynb: Run pathways enrichment analysis on the genes in each subnetowrk determined from the leiden algorithm. This enables biological annotation of modules.  
