import json
import pickle
import time
import torch
from torch import nn
import torch.optim as optim
import numpy as np
import random
import os
import pandas as pd
import datetime
from rdkit import Chem
from torch_geometric.loader import DataLoader
from drugex.training.generators import SequenceRNN
from drugex.data.corpus.vocabulary import VocSmiles
from arguments import al_argparser,logging
from dataprocess import load_dataloader
from finetune_gen import ALFinetune
from metrics import cindex_score,pearson,spearman
from score_func import smiles2affinity
from model import DTA_graph
from dataprocess import DTADataset,process
from splitters import valid_smiles

# change
# ADFRsuite bin path
adfr_path = 'xxx/ADFRsuite-1.0/ADFRsuite_x86_64Linux_1.0/myFolder/bin'
os.environ['PATH'] += os.pathsep + adfr_path

def load_generator(model_dir,gpu):
    generator_path = os.path.join(model_dir, 'chembl_31_smiles_rnn_PT.pkg')
    voc_path = os.path.join(model_dir, 'chembl_31_smiles_rnn_PT.vocab')
    voc = VocSmiles.fromFile(voc_path, encode_frags=False)
    generator = SequenceRNN(voc=voc, is_lstm=True, use_gpus=gpu)
    generator.loadStatesFromFile(generator_path)

    return generator

def load_predictor(model_path,args):

    DTAmodel = DTA_graph(in_dims=65, emb_dim=args.emb_dim, smi_hidden_dim=args.smi_dim)
    state_dict = torch.load(model_path,map_location=lambda storage, loc: storage.cuda(args.device))
    DTAmodel.load_state_dict(state_dict)
    return DTAmodel

def train(trainloader,testloader,model, optimizer,scheduler,criterion, save_path, device,epochs, stop_epoch):
    break_flag = False
    max_ci = 0
    min_loss = float("inf")
    counter = 0
    for epoch in range(epochs):
        if break_flag:
            break
        model.train()
        running_loss = []
        for i, batch in enumerate(trainloader):
            datas = batch.to(device)
            if len(datas)==1:
                continue

            optimizer.zero_grad()
            output, _ = model(datas)

            loss = criterion(output.view(-1), datas.y.view(-1)).float()
            running_loss.append(loss.cpu().detach())

            loss.backward()
            optimizer.step()
            scheduler.step()

        epoch_loss = np.mean(running_loss)
        val_loss, val_cindex, val_pear, val_sp = val(testloader, model, criterion, device)
        val_ci = round(val_cindex.item(),3)

        if max_ci < val_ci or (max_ci==val_ci and min_loss > val_loss):
            max_ci = val_ci
            torch.save(model.state_dict(), save_path[0])
        if min_loss > val_loss:
            min_loss = val_loss
            torch.save(model.state_dict(), save_path[1])
            counter = 0
        else:
            counter = counter + 1
            if counter > stop_epoch:
                print(f"early stop in epoch {epoch}")
                break_flag = True
                break
        print('epoch:{} Average loss: {:.4f} val loss: {:.4f} val ci: {:.4f} val pearson: {:.4f} val sp: {:.4f}'
              .format(epoch, epoch_loss, val_loss, val_cindex, val_pear, val_sp))


def val(dataloader, model, criterion, device):

    model.eval()
    pred_list = []
    label_list = []
    running_loss = []

    for i, batch in enumerate(dataloader):
        datas = batch.to(device)

        with torch.no_grad():
            output, _ = model(datas)
            loss = criterion(output.view(-1), datas.y.view(-1)).float()

        pred_list.append(output.view(-1).detach().cpu().numpy())
        label_list.append(datas.y.view(-1).detach().cpu().numpy())
        running_loss.append(loss.cpu().detach())

    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)
    pred = torch.from_numpy(pred)
    label = torch.from_numpy(label)

    epoch_loss = np.mean(running_loss)
    epoch_cindex = cindex_score(label, pred)
    epoch_pear = pearson(label, pred)
    epoch_sp = spearman(label, pred)


    return epoch_loss, epoch_cindex, epoch_pear, epoch_sp

def testmodel(protein_name,device,args,save_path=None):
    # load data
    if not os.path.exists(os.path.join('data/test',f'{protein_name}_random')):
        os.makedirs(os.path.join('data/test',f'{protein_name}_random','raw'))
        test_path = os.path.join('data/test',f'{protein_name}_test_random.csv')
        testdata = pd.read_csv(test_path)
        test_smiles = testdata['smiles'].tolist()
        test_affinity = testdata['affinity'].tolist()
        test_save_path = os.path.join('data/test',f'{protein_name}_random', 'raw', 'data_test.npy')
        train_save_path = os.path.join('data/test',f'{protein_name}_random', 'raw', 'data_train.npy')

        process(test_smiles, test_affinity, args.max_smi_len, test_save_path)
        process(test_smiles[:10], test_affinity[:10], args.max_smi_len, train_save_path)

    if not os.path.exists(os.path.join('data/test',f'{protein_name}_scaffold')):
        os.makedirs(os.path.join('data/test',f'{protein_name}_scaffold','raw'))
        test_path = os.path.join('data/test',f'{protein_name}_test_scaffold.csv')
        testdata = pd.read_csv(test_path)
        test_smiles = testdata['smiles'].tolist()
        test_affinity = testdata['affinity'].tolist()
        test_save_path = os.path.join('data/test',f'{protein_name}_scaffold', 'raw', 'data_test.npy')
        train_save_path = os.path.join('data/test',f'{protein_name}_scaffold', 'raw', 'data_train.npy')

        process(test_smiles, test_affinity, args.max_smi_len, test_save_path)
        process(test_smiles[:10], test_affinity[:10], args.max_smi_len, train_save_path)


    random_dataset = DTADataset(os.path.join('data/test',f'{protein_name}_random'),train=False,MAX_SMI_LEN=args.max_smi_len)
    scaffold_dataset = DTADataset(os.path.join('data/test',f'{protein_name}_scaffold'),train=False,MAX_SMI_LEN=args.max_smi_len)

    random_loader = DataLoader(dataset=random_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    scaffold_loader = DataLoader(dataset=scaffold_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # build model
    DTAmodel = DTA_graph(in_dims=65, emb_dim=args.emb_dim, smi_hidden_dim=args.smi_dim)
    DTAmodel.to(device)
    criterion = nn.MSELoss()

    # test
    best_model = load_predictor(save_path,args)
    best_model.to(device)

    loss, ci, pear, sp = val(random_loader,best_model,criterion,device)

    logging(
        "RAMDOM TEST Dataset CI = %f, MSE = %f, pearson = %f sp = %f" % (ci, loss, pear, sp), args)

    loss, ci, pear, sp = val(scaffold_loader, best_model, criterion, device)

    logging(
        "SCAFFOLD TEST Dataset CI = %f, MSE = %f, pearson = %f sp = %f" % (ci, loss, pear, sp), args)

def init_labeldata(generater_path=None,data_path=None,protein_name=None,device=0):
    protein_file = os.path.join('data/test', protein_name)
    # change
    # autodock-gpu & gpf.py in Vina path
    autodock = 'xxx/AutoDock-GPU-develop/bin/autodock_gpu_64wi'
    gpf = 'xxx/AutoDock-Vina/example/autodock_scripts/prepare_gpf.py'

    # init training set for gal-dta
    save_path = os.path.join(data_path, 'generate')
    if not os.path.exists(os.path.join(save_path, f'{protein_name}_train.csv')):
        generater = load_generator(generater_path, tuple([device]))
        init_smiles = []
        init_affinity = []

        smiles = generater.generate(num_samples=120)['SMILES']

        valid_smi = []
        for smi in smiles:
            if valid_smiles(smi):
                mol = Chem.MolFromSmiles(smi)
                Chem.SanitizeMol(mol)
                valid_smi.append(Chem.MolToSmiles(mol))
        print(f'valid smiles num: {len(valid_smi)}')

        affinity = smiles2affinity(smiles=valid_smi, protein_name=protein_name, data_dir=protein_file,
                                     gpf=gpf, autodock=autodock)

        init_smiles.extend(valid_smi)
        init_affinity.extend(affinity)

        affi = pd.Series(random_affinity).astype(bool)

        df = pd.DataFrame({
            'smiles': init_smiles,
            'affinity': init_affinity
        })

        df = df[affi]
        if len(df) > 100:
            df = df.head(100)

        save_path = os.path.join(save_path, f'{protein_name}_train.csv')
        df.to_csv(save_path, index=False)

def save_labeleddata(smiles, protein_name, save_dir,num=100):
    protein_file = os.path.join('data/test', protein_name)
    # change
    # autodock-gpu & gpf.py in Vina path
    autodock = 'xxx/AutoDock-GPU-develop/bin/autodock_gpu_64wi'
    gpf = 'xxx/AutoDock-Vina/example/autodock_scripts/prepare_gpf.py'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f'{protein_name}_train.csv')

    affinity = smiles2affinity(smiles=smiles, protein_name=protein_name, data_dir=protein_file, gpf=gpf,
                               autodock=autodock)
    affi = pd.Series(affinity).astype(bool)

    df = pd.DataFrame({
        'smiles': smiles,
        'affinity': affinity
    })

    df = df[affi]
    if len(df)>num:
        df = df.head(num)

    if os.path.exists(save_path):
        df.to_csv(save_path,mode='a',index=False,header=False)
    else:
        df.to_csv(save_path,index=False)

def GenAL(device, args):
    # train predictor in init training set
    logging("================== ITER = %d ===================" % (args.now), args)
    protein_name = args.protein_name

    csv_dir = 'data/generate'
    save_dir = os.path.join(args.model_dir, 'generate',protein_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f'G{protein_name}_{args.now}.pt')

    voc_path = os.path.join(args.model_dir, 'chembl_31_smiles_rnn_PT.vocab')
    voc = VocSmiles.fromFile(voc_path, encode_frags=False)
    traindataset, testdataset, trainloader, testloader,_ = load_dataloader(data_dir=csv_dir,protein_name=protein_name,args=args)

    predictor = DTA_graph(in_dims=65, emb_dim=args.emb_dim, smi_hidden_dim=args.smi_dim)
    predictor.to(device)
    optimizer = optim.Adam(predictor.parameters(), lr=args.lr, eps=1e-7)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=1,last_epoch=-1)
    criterion = nn.MSELoss()

    train(trainloader=trainloader, testloader=testloader, model=predictor,
          optimizer=optimizer, scheduler=scheduler,criterion=criterion, save_path=save_path,
          device=device, epochs=args.epoch, stop_epoch=args.stop_epoch)

    # test
    testmodel(protein_name, device, args, save_path=save_path)

    ftobj = ALFinetune(dataroot='data', model_dir=args.model_dir,pre_path=save_dir, hidden_dims=args.smi_dim,
                       fea_dim=args.hidden_dim,protein_name=protein_name,csv_dir=csv_dir,
                       emb_dim=args.emb_dim, batchsize=args.batch_size, restore_epoch=args.now,
                       device=args.device, epoch=args.gen_epoch, al_iter=args.now+1, smiles_len=args.max_smi_len)

    for i in range(args.now+1,args.al_iter):
        logging("================== ITER = %d ===================" % (i), args)
        start = datetime.datetime.now()
        save_path = os.path.join(save_dir, f'G{protein_name}_{i}.pt')

        # finetune generator
        ftobj.restore_epoch = i-1
        ftobj.al_iter = i

        ftobj.build_model()
        ftobj.build_Score()
        ftobj.build_explorer()
        ftobj.finetune()

        # generate new data for predictor training set
        model_path = os.path.join(args.model_dir,'GAL',protein_name,f'{protein_name}_{i}.pkg')
        new_generator = SequenceRNN(voc=voc,is_lstm=True,use_gpus=tuple([args.device]))
        new_generator.loadStatesFromFile(model_path)

        smiles = new_generator.generate(num_samples=120)['SMILES']

        valid_smi = []
        for smi in smiles:
            if valid_smiles(smi):
                mol = Chem.MolFromSmiles(smi)
                Chem.SanitizeMol(mol)
                valid_smi.append(Chem.MolToSmiles(mol))

        save_labeleddata(valid_smi,protein_name,save_dir=csv_dir,num=100)

        # train predictor
        traindataset, testdataset, trainloader, testloader,_ = load_dataloader(data_dir=csv_dir,
                                                                             protein_name=protein_name,args=args)
        predictor = DTA_graph(in_dims=65, emb_dim=args.emb_dim, smi_hidden_dim=args.smi_dim)
        predictor.to(device)
        optimizer = optim.Adam(predictor.parameters(), lr=args.lr, eps=1e-7)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, last_epoch=-1)

        train(trainloader=trainloader, testloader=testloader, model=predictor,
              optimizer=optimizer, scheduler=scheduler,criterion=criterion, save_path=save_path,
              device=device, epochs=args.epoch, stop_epoch=args.stop_epoch)

        # test predictor
        testmodel(protein_name, device, args, save_path=save_path)

        end = datetime.datetime.now()
        print(f'totally time is {end - start}')
        run_time = end - start
        logging(f"run time = {str(run_time)}", args)


def main(args):
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    GenAL(device,args)

if __name__ == '__main__':
    args = al_argparser()
    timestamp = time.strftime("%Y%m%d_%H")
    args.log_dir = args.log_dir + str(timestamp) + "/"

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logging(str(args), args)

    if args.runseed:
        runseed = args.runseed
        print('Manual runseed: ', runseed)
    else:
        runseed = random.randint(0, 10000)
        print('Random runseed: ', runseed)

    torch.manual_seed(runseed)
    np.random.seed(runseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(runseed)
    args.runseed = runseed
    # first run without init training set
    # init_labeldata(generater_path=args.model_dir, data_path='data',protein_name=args.protein_name, device=args.device)
    main(args)