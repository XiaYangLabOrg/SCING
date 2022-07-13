# SCING
SCING: Single Cell INtegrative Gene regulatory network inference
  
  ---
  ### Downloading the repo  
  ``` 
  $git clone https://github.com/XiaYangLabOrg/SCING.git
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
