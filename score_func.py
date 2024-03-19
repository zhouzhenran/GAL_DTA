import gc
import os.path
import shutil
import numpy as np
import math
import torch
import pandas as pd
import subprocess
import datetime
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from rdkit import DataStructs,Chem
from rdkit.Chem import AllChem
from sklearn.metrics import pairwise_distances
from dataprocess import DTADataset,label_smiles,CHARISOSMISET
import meeko

# autodock - 'xxx/AutoDock-GPU-develop/bin/autodock_gpu_64wi'
# gpf - 'xxx/AutoDock-Vina/example/autodock_scripts/prepare_gpf.py'
delta_g_to_kd = lambda x: math.exp(x / (0.00198720425864083 * 298.15))

def get_pool_embedding(smiles, DTAmodel,mol_dir, protein_name,smiles_len):
    '''
    generative data embedding in finetune process
    mol_dir:data/learn
    '''
    embDim = DTAmodel.hidden_dim
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    train_save_path = os.path.join(mol_dir,f'{protein_name}','raw','data_train.npy')
    test_save_path = os.path.join(mol_dir,f'{protein_name}','raw','data_test.npy')
    if not os.path.exists(os.path.join(mol_dir,f'{protein_name}','raw')):
        os.makedirs(os.path.join(mol_dir,f'{protein_name}','raw'))
    else:
        print('delete data_train.npy')
        os.system(f'rm {train_save_path}')

    if os.path.exists(os.path.join(mol_dir,f'{protein_name}','processed')):
        os.system(f'rm {mol_dir}/{protein_name}/processed/*.pt')

    ligands = []
    proteins = []
    aff = []

    for smi in smiles:
        ligands.append(label_smiles(smi,smiles_len,CHARISOSMISET))
        proteins.append(ligands[-1])
        aff.append(0)

    df = pd.DataFrame(
        {
            'smiles': smiles,
            'smiles_emb': ligands,
            'protein': proteins,
            'affinity': aff
        }
    )

    df_save = df.to_numpy()
    np.save(train_save_path, df_save)
    if not os.path.exists(test_save_path):
        df1 = df.head(2)
        df1 = df1.to_numpy()
        np.save(test_save_path,df1)

    dataset = DTADataset(os.path.join(mol_dir,f'{protein_name}'),train=True,MAX_SMI_LEN=smiles_len)
    dataloader = DataLoader(dataset=dataset,batch_size=1,shuffle=False,num_workers=0)

    embedding = torch.zeros([len(dataset), embDim])
    DTAmodel.to(device)
    DTAmodel.eval()
    for i,batch in enumerate(dataloader):
        data = batch.to(device)

        with torch.no_grad():
            output,emb = DTAmodel(data)
            embedding[i] = emb.data.cpu()

    embedding = embedding.unsqueeze(dim=1)#[n,1,128]

    return embedding


def get_train_embedding(DTAmodel,mol_dir, protein_name, smiles_len):
    embDim = DTAmodel.hidden_dim
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # data/generate/protein_name
    mol_dir = os.path.join(mol_dir,f'{protein_name}')

    dataset = DTADataset(mol_dir,train=True,MAX_SMI_LEN=smiles_len)
    dataloader = DataLoader(dataset=dataset, batch_size=1,shuffle=False, num_workers=0)
    embedding = torch.zeros([len(dataset), embDim])
    DTAmodel.to(device)
    DTAmodel.eval()
    for i, batch in enumerate(dataloader):
        data = batch.to(device)

        with torch.no_grad():
            output, emb = DTAmodel(data)
            embedding[i] = emb.data.cpu()

    embedding = embedding.unsqueeze(dim=1)  # [n,1,128]

    return embedding

def compute_trace(X, X_train, fisher, iterates, lamb=1):
    K = len(X)
    N_train = len(X_train)
    dim = X.shape[-1]
    rank = X.shape[-2]

    currentInv = torch.inverse(lamb * torch.eye(dim).cuda() + iterates.cuda() * N_train / (N_train + K))#[128,128]
    X = X * np.sqrt(K / (N_train + K))
    fisher = fisher.cuda()

    print('compute trace...')

    # check trace with low-rank updates (woodbury identity)
    xt_ = X.cuda()
    innerInv = torch.inverse(torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
    innerInv[torch.where(torch.isinf(innerInv))] = torch.sign(
        innerInv[torch.where(torch.isinf(innerInv))]) * np.finfo('float32').max
    traceEst = torch.diagonal(xt_ @ currentInv @ fisher @ currentInv @ xt_.transpose(1, 2) @ innerInv, dim1=-2,
                              dim2=-1).sum(-1)

    # clear out gpu memory
    xt = xt_.cpu()
    del xt, innerInv
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    traceEst = traceEst.detach().cpu().numpy()
    max_tr = np.max(traceEst)
    min_tr = np.min(traceEst)
    normalized_tr = (traceEst - min_tr) / (max_tr - min_tr)

    return normalized_tr

def compute_firsher_score(xt,x_train,batchsize):
    # get fisher
    print('getting fisher matrix...', flush=True)
    fisher = torch.zeros(xt.shape[-1], xt.shape[-1])
    for i in range(int(np.ceil(len(xt) / batchsize))):
        xt_ = xt[i * batchsize : (i + 1) * batchsize].cuda()
        op = torch.sum(torch.matmul(xt_.transpose(1,2), xt_) / (len(xt)), 0).detach().cpu()
        fisher = fisher + op
        xt_ = xt_.cpu()
        del xt_, op
        torch.cuda.empty_cache()
        gc.collect()

    # fisher = fisher + init
    init = torch.zeros(xt.shape[-1], xt.shape[-1])
    for i in range(int(np.ceil(len(x_train) / batchsize))):
        xt_ = x_train[i * batchsize: (i + 1) * batchsize].cuda()
        op = torch.sum(torch.matmul(xt_.transpose(1, 2), xt_) / (len(x_train)), 0).detach().cpu()
        init = init + op
        fisher = fisher + op
        xt_ = xt_.cpu()
        del xt_, op
        torch.cuda.empty_cache()
        gc.collect()

    scores = compute_trace(X=xt, X_train=x_train, fisher=fisher, iterates=init,lamb=1)
    return scores

def compute_similarity(pool_fps, train_fps):
    sim_scores = []
    for fps in pool_fps:
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fps,train_fps))
        sim_scores.append(sims.max())
    return sim_scores


def smiles2affinity(smiles, protein_name,data_dir,gpf, autodock):
    original_path = os.getcwd()
    os.chdir(data_dir)
    lignds_dir = 'ligands'
    outs_dir = 'outs'
    if not os.path.exists(lignds_dir):
        os.mkdir(lignds_dir)
    if not os.path.exists(outs_dir):
        os.mkdir(outs_dir)
    os.system(f'rm {outs_dir}/*.xml')
    os.system(f'rm {outs_dir}/*.dlg')
    os.system(f'rm -rf {lignds_dir}/*')
    with open('box.txt','r') as f:
        lines = f.readlines()
    box = [float(x) for x in lines[1].split()[1:4]]
    protein_file = protein_name+'.pdbqt'

    for i,smi in enumerate(tqdm(smiles, desc='preparing ligands')):
        sdf_file = f'{lignds_dir}/ligand{i}.sdf'
        ligand_file = f'{lignds_dir}/ligand{i}.pdbqt'
        mol = Chem.MolFromSmiles(smi)

        try:
            mol = AllChem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            print("No conformers found for the molecule.")
            mol = AllChem.AddHs(mol)

        Chem.MolToMolFile(mol,sdf_file)
        os.system(f'mk_prepare_ligand.py -i  {sdf_file} -o {ligand_file}')
        count = 0
        while not os.path.exists(ligand_file) and count < 5:
            os.system(f'mk_prepare_ligand.py -l {sdf_file} -o {ligand_file}')
            count = count+1
        if os.path.exists(ligand_file):
            # get gpf
            os.system(f'pythonsh {gpf} -l {ligand_file} -r {protein_file} -p npts=126,126,126 gridcenter={box[0]},{box[1]},{box[2]}')
            # get maps.fld
            os.system(f'autogrid4 -p {protein_name}.gpf -l {protein_name}.glg')
            # dock
            protein_fld = protein_name + '.maps.fld'
            os.system(f'CUDA_VISIBLE_DEVICES=0 {autodock} -M {protein_fld} -L {ligand_file} -N {outs_dir}/ligand{i}')

    affins = [0 for _ in range(len(smiles))]
    for file in tqdm(os.listdir(outs_dir), desc='extracting binding values'):
        if file.endswith('.dlg') and '0.000   0.000   0.000  0.00  0.00' not in open(f'{outs_dir}/{file}').read():
            affins[int(file.split('ligand')[1].split('.')[0])] = float(
                subprocess.check_output(f"grep 'RANKING' {outs_dir}/{file} | tr -s ' ' | cut -f 5 -d ' ' | head -n 1",
                                        shell=True).decode('utf-8').strip())
    os.chdir(original_path)

    return affins
