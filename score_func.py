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
from dataprocess import DTADataset1
from datahelper import label_smiles,CHARISOSMISET,fp_smiles
import meeko

# autodock - '../ALGAN-dock/programs/AutoDock-GPU-develop/bin/autodock_gpu_64wi'
# gpf - '/home/zhouzhenran/ALDTA/AutoDock-Vina/example/autodock_scripts/prepare_gpf.py'
# protein_file - '1err.pdbqt'
delta_g_to_kd = lambda x: math.exp(x / (0.00198720425864083 * 298.15))

def get_pool_embedding(smiles, DTAmodel,mol_dir, protein_name,smiles_len):
    '''
    生成模型微调时计算生成分子的embedding，type=learn
    微调时生成分子集路径 data/aldata/learn/
    :param DTAmodel: 相互作用预测模型
    :param Molmodel: 分子生成模型
    :param dataroot: 分子数据根路径
    :param batchsize:
    :param protein_name: 靶蛋白名字
    :param name: 用于训练or测试
    :param type: 用于强化学习微调生成模型【learn】or生成模型生成的分子用于训练相互作用模型【generate】
    :return:
    '''
    embDim = DTAmodel.hidden_dim
    device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

    # 已知 给定靶点
    # 待处理 生成分子-转为graph
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
    fps = []
    proteins = []
    aff = []

    for smi in smiles:
        ligands.append(label_smiles(smi,smiles_len,CHARISOSMISET))
        fps.append(fp_smiles(smi))
        proteins.append(ligands[-1])
        aff.append(0)

    df = pd.DataFrame(
        {
            'smiles': smiles,
            'smiles_emb': ligands,
            'fps': fps,
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

    dataset = DTADataset1(os.path.join(mol_dir,f'{protein_name}'),train=True,MAX_SMI_LEN=smiles_len)
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

    # 训练集路径 data/aldata/generate/protein_name
    mol_dir = os.path.join(mol_dir,f'{protein_name}')

    dataset = DTADataset1(mol_dir,train=True,MAX_SMI_LEN=smiles_len)
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
    dim = X.shape[-1]#128
    rank = X.shape[-2]#1

    # torch.inverse-计算矩阵的逆
    # torch.eye(dim)单位矩阵[128,128]
    # 训练集fisher矩阵iterates[128,128]进行缩放
    currentInv = torch.inverse(lamb * torch.eye(dim).cuda() + iterates.cuda() * N_train / (N_train + K))#[128,128]
    # X-[K,1,128]进行缩放
    X = X * np.sqrt(K / (N_train + K))
    fisher = fisher.cuda()

    print('compute trace...')

    # check trace with low-rank updates (woodbury identity)
    xt_ = X.cuda()#[k,1,128]
    # [k,1,1]+[1,128]x[k,128,128]x[k,128,1]=[k,1,1]
    innerInv = torch.inverse(torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
    # 避免溢出，将矩阵中无穷大的元素替换成同符号的浮点数最大值
    innerInv[torch.where(torch.isinf(innerInv))] = torch.sign(
        innerInv[torch.where(torch.isinf(innerInv))]) * np.finfo('float32').max
    # [k,1,128]x[128,128]x[128,128]x[128,128]x[k,128,1]x[1,1]=[k,1,1]
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
    # return traceEst

def compute_firsher_score(xt,x_train,batchsize):
    # get fisher
    print('getting fisher matrix...', flush=True)
    # 先计算pool data fisher matrix - fisher
    fisher = torch.zeros(xt.shape[-1], xt.shape[-1])#[128,128]
    for i in range(int(np.ceil(len(xt) / batchsize))):
        xt_ = xt[i * batchsize : (i + 1) * batchsize].cuda()
        op = torch.sum(torch.matmul(xt_.transpose(1,2), xt_) / (len(xt)), 0).detach().cpu()
        fisher = fisher + op
        xt_ = xt_.cpu()
        del xt_, op
        torch.cuda.empty_cache()
        gc.collect()

    # 计算 train data fisher matrix - init
    # fisher = fisher + init 总数据的fisher matrix
    init = torch.zeros(xt.shape[-1], xt.shape[-1])#[128,128]
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
    print("smiles num: {}".format(len(smiles)))
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

def out2affinity(smiles, protein_name,data_dir,gpf, autodock):
    original_path = os.getcwd()
    os.chdir(data_dir)
    outs_dir = 'outs'

    print("smiles num: {}".format(len(smiles)))

    affins = [0 for _ in range(len(smiles))]
    for file in tqdm(os.listdir(outs_dir), desc='extracting binding values'):
        if file.endswith('.dlg') and '0.000   0.000   0.000  0.00  0.00' not in open(f'{outs_dir}/{file}').read():
            affins[int(file.split('ligand')[1].split('.')[0])] = float(
                subprocess.check_output(f"grep 'RANKING' {outs_dir}/{file} | tr -s ' ' | cut -f 5 -d ' ' | head -n 1",
                                        shell=True).decode('utf-8').strip())
    os.chdir(original_path)

    return affins
