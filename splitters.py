import pandas as pd
import random
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import Chem
import argparse
import os
from score_func import smiles2affinity

def valid_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
        return True
    except:
        return False
# splitter function

def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold

def scaffold_split(smiles_list, num=3,max_num=500):
    """
    Adapted from https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
    Split dataset by Bemis-Murcko scaffolds
    """
    # create dict of the form {scaffold_i: [idx1, idx....]}
    all_scaffolds = {}
    for i, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles, include_chirality=True)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    # sort from largest to smallest sets
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]

    # get test indices
    test_idx = []
    for scaffold_set in all_scaffold_sets:
        if len(scaffold_set) < num:
            test_idx.extend(scaffold_set)
        else:
            test_idx.extend((scaffold_set[:num]))
        if len(test_idx) >= max_num:
            break

    test_smiles = [smiles_list[i] for i in test_idx]

    return test_smiles

def scaffold_num(smiles_list):
    # create dict of the form {scaffold_i: [idx1, idx....]}
    all_scaffolds = {}
    for i, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles, include_chirality=True)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    # sort from largest to smallest sets
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    # print(all_scaffold_sets)
    for i in range(300):
        print(len(all_scaffold_sets[i]))
    print(f"scaffold num:{len(all_scaffold_sets)}")

def random_split(smiles_list, seed=0,num=500):

    num_mols = len(smiles_list)
    random.seed(seed)
    all_idx = list(range(num_mols))
    random.shuffle(all_idx)

    test_idx = all_idx[:num*2]

    test_smiles = [smiles_list[i] for i in test_idx]

    path = os.path.join('data/test','P51449_test_random.csv')
    smiles1 = pd.read_csv(path)
    smiles1 = smiles1['smiles']
    path = os.path.join('data/test', 'P51449_test_scaffold.csv')
    smiles2 = pd.read_csv(path)
    smiles2 = smiles2['smiles']

    train_smiles = []
    for smi in test_smiles:
        if smi not in smiles1 and smi not in smiles2:
            train_smiles.append(smi)
        if len(train_smiles)>=(num+100):
            break

    return train_smiles

def generate_testdataset(protein_name,num=5,split_type='random'):
    save_path = 'data/test'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data_dir = 'data/chembl.smi'
    with open(data_dir,'r') as f:
        smiles_list = f.read().splitlines()

    if split_type == 'random':
        smiles = random_split(smiles_list,seed=24,num=num)
    else:
        smiles = scaffold_split(smiles_list,num=2,max_num=num)

    print(f'test len: {len(smiles)}')

    valid_smi = []
    for smi in smiles:
        if valid_smiles(smi):
            mol = Chem.MolFromSmiles(smi)
            Chem.SanitizeMol(mol)
            valid_smi.append(Chem.MolToSmiles(mol))
    print(f'valid smiles num: {len(valid_smi)}')

    protein_file = os.path.join('data/test', protein_name)
    # change
    # autodock-gpu & gpf.py in Vina path
    autodock = 'xxx/AutoDock-GPU-develop/bin/autodock_gpu_64wi'
    gpf = 'xxx/AutoDock-Vina/example/autodock_scripts/prepare_gpf.py'

    affinity = smiles2affinity(smiles=valid_smi, protein_name=protein_name, data_dir=protein_file, gpf=gpf,
                               autodock=autodock)

    df = pd.DataFrame({
        'smiles': valid_smi,
        'affinity': affinity
    })

    save_path = os.path.join(save_path, f'{protein_name}_test_{split_type}.csv')
    df.to_csv(save_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--protein_name', type=str, default='P51449')
    parser.add_argument('--num', type=int, default=2000)
    parser.add_argument('--split_type', type=str, default='random',help='random or scaffold')
    args = parser.parse_args()
    generate_testdataset(args.protein_name,num=args.num,split_type=args.split_type)