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
Then run `splitters.py` like:
```
python splitters.py --protein_name P51449 --num 2000 --split_type random
```
Explanation of parameters  
* --protein_name: protein name, which needs to be consistent with the name of the `data\test\xxx` folder  
* --num: number of test set data.  
* --split_type: the method of filtering data. You can choose random or scaffold.
## Usage
```
python train.py --protein_name P51449 --al_iter 40 --num_sample 100 --gen_epoch 10 --epoch 1000 --log_dir logs/
```
Explanation of parameters  
* --protein_name: protein name, which needs to be consistent with the name of the `data\test\xxx` folder  
* --al_iter: number of data augmentation phase iterations  
* --num_sample: number of data (for predictor training) generated by the generator in each epoch of the data augmentation phase  
* --gen_epoch: number of iterations for generator fine-tuning.  
* --epoch: number of predictor training iterations.  
* --log_dir: the location where the metrics output is saved.

## Output
After executing the train.py file, at each epoch of data augmentation, the data generated by the generator (for predictor training)
will be saved in `data\generate\xxx(your protein name)_train.csv`.  
The output file will be saved in the folder set by the parameter `--log_dir`.
The content of the file includes all the parameter settings, test results (MSE, CI, Pearson and Sp) of the trained predictor on both test sets in each epoch of data augmentation.