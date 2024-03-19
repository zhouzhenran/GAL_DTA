# Effective Drug-Target Affinity Prediction via Generative Active Learning
This is the implementation for Effective Drug-Target Affinity Prediction via Generative Active Learning
## Requirement
drugex==3.4.4  
meeko==0.4.0  
networkx==2.6.3  
numpy==1.21.6  
pandas==1.3.5  
scikit_learn==1.0.2  
seaborn==0.13.2  
torch==1.12.1  
torch_geometric==2.2.0  
tqdm==4.63.1  
rdkit  
ADFRsuite-1.0  
AutoDock-Vina (https://github.com/ccsb-scripps/AutoDock-Vina)  
AutoDock-GPU (https://github.com/ccsb-scripps/AutoDock-GPU)  
  
After installing ADFRsuite, AutoDock-Vina and AutoDock-GPU, you need to change `adfr_path`, `autodock` and `gpf` in `train.py` to the correct paths.
## Data
The `data\test` folder provides `.pdbqt` files and test sets for the four target proteins used in the paper.  
If you want to generate test set for other protein, you need to save the `.pdbqt` file for that protein as well as the box size and coordinates (`box.txt`) in `data\test\xxx(your protein name)` folder.
Then run `split.py` like:
```
python splitters.py --protein_name P51449 --num 2000 --split_type random
```

## Usage
```
python train.py --protein_name P51449
```