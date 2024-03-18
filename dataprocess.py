import os
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
import numpy as np
import time
from copy import deepcopy
from rdkit import Chem
from rdkit.Chem import AllChem
import networkx as nx
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
import math
import pandas as pd
from tqdm import tqdm
from datahelper import *

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return features, edge_index

class DTADataset1(InMemoryDataset):

    def __init__(self, root, train=True, transform=None, pre_transform=None, pre_filter=None, MAX_SMI_LEN=442):
        self.MAX_SMI_LEN = MAX_SMI_LEN
        super().__init__(root, transform, pre_transform, pre_filter)
        if train:
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.data, self.slices = torch.load(self.processed_paths[1])

    @property
    def raw_file_names(self):
        return ['data_train.npy', 'data_test.npy']

    @property
    def processed_file_names(self):
        return ['processed_data_train.pt', 'processed_data_test.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass


    def process_data(self, data_path, graph_dict):
        # df = pd.read_csv(data_path)
        df = np.load(data_path,allow_pickle=True)
        df = pd.DataFrame(df,columns=['smiles','smiles_emb','fps','protein','affinity'])

        data_list = []
        for i, row in df.iterrows():
            smi = row['smiles']
            sequence = row['protein'].tolist()
            label = row['affinity']

            x, edge_index = graph_dict[smi]
            fps = row['fps']
            smi_emb = row['smiles_emb']

            # Get Labels
            try:
                data = DATA.Data(
                    x=torch.FloatTensor(x),
                    edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                    y=torch.FloatTensor([label]),
                    target=torch.FloatTensor([sequence]),
                    fps = torch.FloatTensor([fps]),
                    smi_emb = torch.LongTensor([smi_emb])
                )
            except:
                    print("unable to process: ", smi)

            data_list.append(data)

        return data_list

    def process(self):
        # df_train = pd.read_csv(self.raw_paths[0])
        # df_test = pd.read_csv(self.raw_paths[1])
        df_train = np.load(self.raw_paths[0],allow_pickle=True)
        df_test = np.load(self.raw_paths[1],allow_pickle=True)
        df_train = pd.DataFrame(df_train,columns=['smiles','smiles_emb','fps','protein','affinity'])
        df_test = pd.DataFrame(df_test,columns=['smiles','smiles_emb','fps','protein','affinity'])
        df = pd.concat([df_train, df_test])

        smiles = df['smiles'].unique()
        graph_dict = dict()

        for smile in tqdm(smiles, total=len(smiles)):
            g = smile_to_graph(smile)
            graph_dict[smile] = g

        train_list = self.process_data(self.raw_paths[0], graph_dict)
        test_list = self.process_data(self.raw_paths[1], graph_dict)

        if self.pre_filter is not None:
            train_list = [train for train in train_list if self.pre_filter(train)]
            test_list = [test for test in test_list if self.pre_filter(test)]

        if self.pre_transform is not None:
            train_list = [self.pre_transform(train) for train in train_list]
            test_list = [self.pre_transform(test) for test in test_list]

        print('Graph construction done. Saving to file.')

        data, slices = self.collate(train_list)
        # save preprocessed train data:
        torch.save((data, slices), self.processed_paths[0])

        data, slices = self.collate(test_list)
        # save preprocessed test data:
        torch.save((data, slices), self.processed_paths[1])

def dataprocess(args):

    dataset = DataSet(
        f_path = args.dataset,
        setting = args.problem_type,
        smilen = args.max_smi_len # 85,100,200,200
    )

    XD,XT,Y,SMILES,FPS = dataset.parse_data()

    XD = np.array(XD)
    XT = np.asarray(XT)
    Y = np.asarray(Y)
    FPS = np.asarray(FPS)

    label_row_inds, label_col_inds = np.where(
        np.isnan(Y) == False)  # basically finds the point address of affinity [x,y]

    test_set, outer_train_sets = dataset.read_sets()

    return XD, XT, Y, SMILES, FPS, label_row_inds, label_col_inds, test_set, outer_train_sets

def prepare_interaction_pairs(XD, XT, Y, SMILES,FPS, rows, cols, flag=True):
    ligands = []
    targets = []
    smiles = []
    affinity = []
    fps = []

    for pair_ind in range(len(rows)):
        ligand = XD[rows[pair_ind]]
        ligands.append(ligand)

        target = XT[cols[pair_ind]]
        targets.append(target)

        smi = SMILES[rows[pair_ind]]
        smiles.append(smi)

        fp = FPS[rows[pair_ind]]
        fps.append(fp)

        affinity.append(Y[rows[pair_ind], cols[pair_ind]])

    if flag:
        ligand_data = np.stack(ligands) # [n,85]
        target_data = np.stack(targets) # [n,442]
        fps_data = np.stack(fps)
    else:
        ligand_data = ligands  # [n,85]
        target_data = targets  # [n,442]
        fps_data = fps

    return ligand_data, target_data, smiles,fps_data, affinity

def split_by_protein(data_path):
    save_dir = 'data/aldata/test'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    proteins = json.load(open(data_path + "proteins.txt"), object_pairs_hook=OrderedDict)
    ligands = json.load(open(data_path + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    Y = pickle.load(open(data_path + "Y", "rb"), encoding='latin1')

    XTN = []
    XT = []
    XD = []

    for t in proteins.keys():
        XTN.append(t)
        XT.append(proteins[t])
    for d in ligands.keys():
        XD.append(ligands[d])

    rows, cols = np.where(np.isnan(Y) == False)

    ligands = []
    targets = []
    targets_name = []
    affinity = []

    for pair_ind in range(len(rows)):
        ligand = XD[rows[pair_ind]]
        ligands.append(ligand)

        target = XT[cols[pair_ind]]
        targets.append(target)

        target_name = XTN[cols[pair_ind]]
        targets_name.append(target_name)

        affinity.append(Y[rows[pair_ind], cols[pair_ind]])

    df = pd.DataFrame({
        'protein': targets_name,
        'sequence': targets,
        'smiles': ligands,
        'affinity': affinity
    })

    groups = df.groupby('protein')
    protein_counts = []

    for name, group in groups:
        count = len(group)
        protein_counts.append([name, count])

        save_path = os.path.join(save_dir, str(name))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, f'{name}.csv')
        group[['smiles', 'affinity']].to_csv(save_path, index=False)

    counts_df = pd.DataFrame(protein_counts, columns=['protein', 'count'])

    save_len = os.path.join(save_dir,'protein.csv')
    counts_df.to_csv(save_len,index=False)
    print(counts_df)

def process(smiles,affinity,max_smi_len,save_path):
    smiles_save = []
    ligands = []
    fps = []
    aff = []
    proteins = []
    for i in range(len(smiles)):
        if not math.isnan(affinity[i]) and affinity[i] < 0:
            smiles_save.append(smiles[i])
            aff.append(affinity[i])
            ligands.append(label_smiles(smiles[i], max_smi_len, CHARISOSMISET))
            fps.append(fp_smiles(smiles[i]))
            proteins.append(ligands[-1])
        else:
            print(f"idx :{i} smiles: {smiles[i]} aff: {affinity[i]}")

    df = pd.DataFrame(
        {
            'smiles': smiles_save,
            'smiles_emb': ligands,
            'fps': fps,
            'protein': proteins,
            'affinity': aff
        }
    )

    df = df.to_numpy()

    np.save(save_path,df)

def load_dataloader(data_dir,protein_name,args,num_iter=0):
    # load data
    train_path = os.path.join(data_dir, f'{protein_name}_train.csv')
    if 'P51449' in protein_name:
        test_path = os.path.join('data/aldata/zinc', f'P51449_test_scaffold.csv')
    elif 'O00329' in protein_name:
        test_path = os.path.join('data/aldata/zinc', f'O00329_test_scaffold.csv')
    elif 'P10721' in protein_name:
        test_path = os.path.join('data/aldata/zinc', f'P10721_test_scaffold.csv')
    elif 'P36888' in protein_name:
        test_path = os.path.join('data/aldata/zinc', f'P36888_test_scaffold.csv')

    traindata = pd.read_csv(train_path)
    testdata = pd.read_csv(test_path)

    train_smiles = traindata['smiles'].tolist()
    train_affinity = traindata['affinity'].tolist()

    test_smiles = testdata['smiles'].tolist()
    test_affinity = testdata['affinity'].tolist()
    if num_iter>0:
        train_smiles = train_smiles[:num_iter]
        train_affinity = train_affinity[:num_iter]

    train_save_path = os.path.join(data_dir,protein_name,'raw','data_train.npy')
    test_save_path = os.path.join(data_dir,protein_name,'raw','data_test.npy')
    if not os.path.exists(os.path.join(data_dir,protein_name,'raw')):
        os.makedirs(os.path.join(data_dir,protein_name,'raw'))

    # dataprocess
    process(train_smiles,train_affinity,args.max_smi_len,train_save_path)

    process(test_smiles,test_affinity,args.max_smi_len,test_save_path)

    if os.path.exists(os.path.join(data_dir,protein_name,'processed')):
        os.system(f'rm {data_dir}/{protein_name}/processed/*.pt')

    traindataset = DTADataset1(os.path.join(data_dir,protein_name),train=True,MAX_SMI_LEN=args.max_smi_len)
    testdataset = DTADataset1(os.path.join(data_dir,protein_name),train=False,MAX_SMI_LEN=args.max_smi_len)

    trainloader = DataLoader(dataset=traindataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(dataset=testdataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    return traindataset,testdataset,trainloader,testloader,train_smiles


