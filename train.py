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
from metrics import cindex_score,get_cindex,get_rm2
from score_func import smiles2affinity
from model import DTA1_graph
from dataprocess import DTADataset1,process
from splitters import random_split,valid_smiles
# import wandb
# os.environ["WANDB_API_KEY"] = 'KEY'
# os.environ["WANDB_MODE"] = "offline"

adfr_path = '/home/zhouzhenran/ALGAN/ALGAN-dock/programs/ADFRsuite-1.0/ADFRsuite_x86_64Linux_1.0/myFolder/bin'
os.environ['PATH'] += os.pathsep + adfr_path
# protein_names = ['P51449','P36888','P10721','P00918','P24941','Q8K4Z4','O00329','P31389','P42345','Q05655']



def load_generator(model_dir,gpu):
    generator_path = os.path.join(model_dir, 'chembl_31_smiles_rnn_PT.pkg')
    voc_path = os.path.join(model_dir, 'chembl_31_smiles_rnn_PT.vocab')
    voc = VocSmiles.fromFile(voc_path, encode_frags=False)
    generator = SequenceRNN(voc=voc, is_lstm=True, use_gpus=gpu)
    generator.loadStatesFromFile(generator_path)

    return generator

def load_predictor(model_path,args):

    DTAmodel = DTA1_graph(in_dims=65, emb_dim=args.emb_dim,hidden_dim=args.hidden_dim, smi_hidden_dim=args.smi_dim)
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
        val_loss, val_cindex, val_r2, val_ci2 = val(testloader, model, criterion, device)
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
        print('epoch:{} Average loss: {:.4f} val loss: {:.4f} val ci: {:.4f} val ci2: {:.4f}'
              .format(epoch, epoch_loss, val_loss, val_cindex, val_ci2))


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
    avg_ci = get_cindex(label, pred)
    pred = torch.from_numpy(pred)
    label = torch.from_numpy(label)
    epoch_cindex = cindex_score(label, pred)
    epoch_loss = np.mean(running_loss)
    epoch_r2 = get_rm2(label, pred)


    return epoch_loss, epoch_cindex, epoch_r2, avg_ci

def testmodel(protein_name,device,args,save_path=None):
    # load data
    if 'P51449' in protein_name:
        protein_name = 'P51449'
    elif 'O00329' in protein_name:
        protein_name = 'O00329'
    elif 'P10721' in protein_name:
        protein_name = 'P10721'
    elif 'P36888' in protein_name:
        protein_name = 'P36888'
    if not os.path.exists(os.path.join('data/aldata/zinc',f'{protein_name}_ra')):
        os.makedirs(os.path.join('data/aldata/zinc',f'{protein_name}_ra','raw'))
        test_path = os.path.join('data/aldata/zinc',f'{protein_name}_test_random.csv')
        testdata = pd.read_csv(test_path)
        test_smiles = testdata['smiles'].tolist()
        test_affinity = testdata['affinity'].tolist()
        test_save_path = os.path.join('data/aldata/zinc',f'{protein_name}_ra', 'raw', 'data_test.npy')
        train_save_path = os.path.join('data/aldata/zinc',f'{protein_name}_ra', 'raw', 'data_train.npy')

        process(test_smiles, test_affinity, args.max_smi_len, test_save_path)
        process(test_smiles[:10], test_affinity[:10], args.max_smi_len, train_save_path)

    if not os.path.exists(os.path.join('data/aldata/zinc',f'{protein_name}_sc')):
        os.makedirs(os.path.join('data/aldata/zinc',f'{protein_name}_sc','raw'))
        test_path = os.path.join('data/aldata/zinc',f'{protein_name}_test_scaffold.csv')
        testdata = pd.read_csv(test_path)
        test_smiles = testdata['smiles'].tolist()
        test_affinity = testdata['affinity'].tolist()
        test_save_path = os.path.join('data/aldata/zinc',f'{protein_name}_sc', 'raw', 'data_test.npy')
        train_save_path = os.path.join('data/aldata/zinc',f'{protein_name}_sc', 'raw', 'data_train.npy')

        process(test_smiles, test_affinity, args.max_smi_len, test_save_path)
        process(test_smiles[:10], test_affinity[:10], args.max_smi_len, train_save_path)


    test2dataset = DTADataset1(os.path.join('data/aldata/zinc',f'{protein_name}_ra'),train=False,MAX_SMI_LEN=args.max_smi_len)
    test3dataset = DTADataset1(os.path.join('data/aldata/zinc',f'{protein_name}_sc'),train=False,MAX_SMI_LEN=args.max_smi_len)

    test2loader = DataLoader(dataset=test2dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test3loader = DataLoader(dataset=test3dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # build model
    DTAmodel = DTA1_graph(in_dims=65, emb_dim=args.emb_dim,hidden_dim=args.hidden_dim, smi_hidden_dim=args.smi_dim)
    DTAmodel.to(device)
    criterion = nn.MSELoss()

    # test
    best_model = load_predictor(save_path[0],args)
    best_model.to(device)

    loss, ci, r2, ci2 = val(test2loader,best_model,criterion,device)

    logging(
        "BEST CI ZINC RAMDOM Dataset CI = %f, MSE = %f, r2m = %f ci = %f" % (ci, loss, r2, ci2), args)

    loss, ci, r2, ci2 = val(test3loader, best_model, criterion, device)

    logging(
        "BEST CI ZINC SCAFFOLD Dataset CI = %f, MSE = %f, r2m = %f ci = %f" % (ci, loss, r2, ci2), args)

    best_model = load_predictor(save_path[1], args)
    best_model.to(device)

    loss, ci, r2, ci2 = val(test2loader, best_model, criterion, device)

    logging(
        "BEST MSE ZINC RAMDOM Dataset CI = %f, MSE = %f, r2m = %f ci = %f" % (ci, loss, r2, ci2), args)

    loss, ci, r2, ci2 = val(test3loader, best_model, criterion, device)

    logging(
        "BEST MSE ZINC SCAFFOLD Dataset CI = %f, MSE = %f, r2m = %f ci = %f" % (ci, loss, r2, ci2), args)

def init_labeldata(generater_path=None,data_path=None,protein_name=None,device=1,flag=1):
    '''

    :param generater_path:
    :param data_path:
    :param protein_name:
    :param device:
    :param flag: 0-random generate data; 1-chembl data; 2-other protein generate data;3-other protein chembl data; 4-other protein test data
    :return:
    '''
    protein_file = os.path.join('data/aldata/test', protein_name)
    autodock = '/home/zhouzhenran/ALGAN/ALGAN-dock/programs/AutoDock-GPU-develop/bin/autodock_gpu_64wi'
    gpf = '/home/zhouzhenran/ALDTA/AutoDock-Vina/example/autodock_scripts/prepare_gpf.py'

    if flag==0:
        # 随机生成数据集
        save_path = os.path.join(data_path, 'random')
        if not os.path.exists(os.path.join(save_path, f'{protein_name}_train_iter.csv')):
            generater = load_generator(generater_path, tuple([device]))
            random_smiles = []
            random_affinity = []
            for i in range(10):
                smiles_1 = generater.generate(num_samples=120)['SMILES']

                valid_smi = []
                for smi in smiles_1:
                    if valid_smiles(smi):
                        mol = Chem.MolFromSmiles(smi)
                        Chem.SanitizeMol(mol)
                        valid_smi.append(Chem.MolToSmiles(mol))
                print(f'valid smiles num: {len(valid_smi)}')

                affinity_1 = smiles2affinity(smiles=valid_smi[:110], protein_name=protein_name, data_dir=protein_file,
                                             gpf=gpf, autodock=autodock)

                random_smiles.extend(valid_smi[:110])
                random_affinity.extend(affinity_1)

            affi = pd.Series(random_affinity).astype(bool)

            df = pd.DataFrame({
                'smiles': random_smiles,
                'affinity': random_affinity
            })

            df = df[affi]
            if len(df) > 1000:
                df = df.head(1000)

            save_path = os.path.join(save_path, f'{protein_name}_train_iter.csv')
            df.to_csv(save_path, index=False)
    elif flag==1:
        # 真实随机训练集
        save_path = os.path.join(data_path, 'zinc')
        if not os.path.exists(os.path.join(save_path, f'{protein_name}_train.csv')):
            data_dir = 'data/aldata/chembl.smi'
            with open(data_dir, 'r') as f:
                smiles_list = f.read().splitlines()
            smiles = random_split(smiles_list,seed=args.runseed,num=1000)

            valid_smi = []
            for smi in smiles:
                if valid_smiles(smi):
                    mol = Chem.MolFromSmiles(smi)
                    Chem.SanitizeMol(mol)
                    valid_smi.append(Chem.MolToSmiles(mol))
            print(f'valid smiles num: {len(valid_smi)}')

            affinity = smiles2affinity(smiles=valid_smi, protein_name=protein_name, data_dir=protein_file, gpf=gpf,
                                       autodock=autodock)

            affi = pd.Series(affinity).astype(bool)
            df = pd.DataFrame({
                'smiles': valid_smi,
                'affinity': affinity
            })
            df = df[affi]
            if len(df) > 1000:
                df = df.head(1000)

            save_path = os.path.join(save_path, f'{protein_name}_train.csv')
            df.to_csv(save_path, index=False)
    elif flag==2:
        # 其他靶蛋白随机数据训练集
        save_path = os.path.join(data_path, 'random')
        if not os.path.exists(os.path.join(save_path, f'{protein_name}_train_iter.csv')) or 1:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if protein_name[-2] != '_':
                valid_smi = pd.read_csv('data/aldata/random/P51449_train_iter.csv')['smiles']
            else:
                print(f"load P51449_{protein_name[-1]}_train.csv")
                valid_smi = pd.read_csv(f'data/aldata/random/P51449_{protein_name[-1]}_train_iter.csv')['smiles']

            valid_smi = valid_smi.tolist()[3001:]

            affinity = smiles2affinity(smiles=valid_smi, protein_name=protein_name, data_dir=protein_file, gpf=gpf,
                                       autodock=autodock)

            affi = pd.Series(affinity).astype(bool)

            df = pd.DataFrame({
                'smiles': valid_smi,
                'affinity': affinity
            })

            df = df[affi]

            save_path = os.path.join(save_path, f'{protein_name}_train_iter.csv')
            if os.path.exists(save_path):
                df.to_csv(save_path,mode='a',index=False,header=False)
            else:
                df.to_csv(save_path, index=False)
    elif flag==3:
        # 其他靶蛋白真实数据训练集
        save_path = os.path.join(data_path, 'zinc')
        if not os.path.exists(os.path.join(save_path, f'{protein_name}_train.csv')) or 1:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if protein_name[-2] != '_':
                valid_smi = pd.read_csv('data/aldata/zinc/P51449_train.csv')['smiles']
            else:
                print(f"load P51449_{protein_name[-1]}_train.csv")
                valid_smi = pd.read_csv(f'data/aldata/zinc/P51449_{protein_name[-1]}_train.csv')[
                    'smiles']
            valid_smi = valid_smi.tolist()[1001:3001]

            affinity = smiles2affinity(smiles=valid_smi, protein_name=protein_name, data_dir=protein_file, gpf=gpf,
                                       autodock=autodock)

            affi = pd.Series(affinity).astype(bool)

            df = pd.DataFrame({
                'smiles': valid_smi,
                'affinity': affinity
            })

            df = df[affi]

            save_path = os.path.join(save_path, f'{protein_name}_train.csv')
            if os.path.exists(save_path):
                df.to_csv(save_path, mode='a', index=False, header=False)
            else:
                df.to_csv(save_path, index=False)
    elif flag==4:
        # 其他靶蛋白测试集
        save_path = os.path.join(data_path, 'zinc')
        if not os.path.exists(os.path.join(save_path, f'{protein_name}_test_random.csv')):
            valid_smi = pd.read_csv('data/aldata/zinc/P51449_test_random.csv')['smiles']

            affinity = smiles2affinity(smiles=valid_smi, protein_name=protein_name, data_dir=protein_file, gpf=gpf,
                                       autodock=autodock)

            affi = pd.Series(affinity).astype(bool)

            df = pd.DataFrame({
                'smiles': valid_smi,
                'affinity': affinity
            })

            df = df[affi]

            save_path1 = os.path.join(save_path, f'{protein_name}_test_random.csv')
            df.to_csv(save_path1, index=False)
        if not os.path.exists(os.path.join(save_path, f'{protein_name}_test_scaffold.csv')):

            valid_smi = pd.read_csv('data/aldata/zinc/P51449_test_scaffold.csv')['smiles']

            affinity = smiles2affinity(smiles=valid_smi, protein_name=protein_name, data_dir=protein_file, gpf=gpf,
                                       autodock=autodock)

            affi = pd.Series(affinity).astype(bool)

            df = pd.DataFrame({
                'smiles': valid_smi,
                'affinity': affinity
            })

            df = df[affi]

            save_path2 = os.path.join(save_path, f'{protein_name}_test_scaffold.csv')
            df.to_csv(save_path2, index=False)

    else:
        print('该靶点已有随机数据集')


def save_labeleddata(smiles, protein_name, save_dir,num=300):
    protein_file = os.path.join('data/aldata/test', protein_name)
    autodock = '/home/zhouzhenran/ALGAN/ALGAN-dock/programs/AutoDock-GPU-develop/bin/autodock_gpu_64wi'
    gpf = '/home/zhouzhenran/ALDTA/AutoDock-Vina/example/autodock_scripts/prepare_gpf.py'

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
    '''
    流程：
    1、设置参数
    2、创建MyPredict，ALFinetune对象，设置参数
    3、开始主动学习迭代
        3.1、根据生成器生成n个分子，对接打分，追加保存在aldata/generate/raw/data_train.csv中
        3.2、提取分子特征
        3.3、训练预测器
        3.4、测试预测器
        3.5、根据预测器来微调生成器
        3.6、重复3.1-3.4直到达到迭代次数

    :param args:
    :return:
    '''

    # al iter 0
    # 生成初始训练集训练预测模型
    logging("================== ITER = %d ===================" % (args.now), args)
    protein_name = args.protein_name
    if args.al_iter >= 10:
        csv_dir = 'data/aldata/generate/iter'
        save_dir = os.path.join(args.model_dir, 'generate','iter',protein_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = [os.path.join(save_dir, f'G{protein_name}_ci_{args.now}.pt'),os.path.join(save_dir, f'G{protein_name}_mse_{args.now}.pt')]
    else:
        csv_dir = 'data/aldata/generate'
        save_dir = os.path.join(args.model_dir, 'generate',protein_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = [os.path.join(save_dir, f'G{protein_name}_ci_{args.now}.pt'),os.path.join(save_dir, f'G{protein_name}_mse_{args.now}.pt')]

    voc_path = os.path.join(args.model_dir, 'chembl_31_smiles_rnn_PT.vocab')
    voc = VocSmiles.fromFile(voc_path, encode_frags=False)
    traindataset, testdataset, trainloader, testloader,_ = load_dataloader(data_dir=csv_dir,protein_name=protein_name,args=args)

    logging("data num = %d" % (len(traindataset)), args)

    predictor = DTA1_graph(in_dims=65, emb_dim=args.emb_dim,hidden_dim=args.hidden_dim, smi_hidden_dim=args.smi_dim)
    predictor.to(device)
    optimizer = optim.Adam(predictor.parameters(), lr=args.lr, eps=1e-7)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=1,last_epoch=-1)
    criterion = nn.MSELoss()

    train(trainloader=trainloader, testloader=testloader, model=predictor,
          optimizer=optimizer, scheduler=scheduler,criterion=criterion, save_path=save_path,
          device=device, epochs=args.epoch, stop_epoch=args.stop_epoch)

    # test
    testmodel(protein_name, device, args, save_path=save_path)

    ftobj = ALFinetune(dataroot='data/aldata', model_dir=args.model_dir,pre_path=save_dir, hidden_dims=args.smi_dim,
                       fea_dim=args.hidden_dim,protein_name=protein_name,csv_dir=csv_dir,
                       emb_dim=args.emb_dim, batchsize=args.batch_size, restore_epoch=args.now,
                       device=args.device, epoch=args.gen_epoch, al_iter=args.now+1, smiles_len=args.max_smi_len)

    for i in range(args.now+1,args.al_iter):
        logging("================== ITER = %d ===================" % (i), args)
        start = datetime.datetime.now()
        save_path = [os.path.join(save_dir, f'G{protein_name}_ci_{i}.pt'),os.path.join(save_dir, f'G{protein_name}_mse_{i}.pt')]

        # finetune
        ftobj.restore_epoch = i-1
        ftobj.al_iter = i

        ftobj.build_model()
        if i<2:
            ftobj.build_Score(args.gen_score[0])
        elif i<5:
            ftobj.build_Score(args.gen_score[1])
        else:
            ftobj.build_Score(args.gen_score[2])
        ftobj.build_explorer()
        score1 = ftobj.test()
        print(f'before finetune score:{score1}')
        ftobj.finetune()
        score = ftobj.test()
        print(f'before finetune score:{score1}')
        print(f'after finetune score:{score}')
        logging("before finetune score = %f" % (score1), args)
        logging("after finetune score = %f" % (score), args)

        # generate new data for predictor train
        model_path = os.path.join(args.model_dir,'RL',protein_name,f'{protein_name}_{i}.pkg')
        new_generator = SequenceRNN(voc=voc,is_lstm=True,use_gpus=tuple([args.device]))
        new_generator.loadStatesFromFile(model_path)

        if args.al_iter == 4:
            smiles = new_generator.generate(num_samples=400)['SMILES']
            num = 300
        else:
            smiles = new_generator.generate(num_samples=120)['SMILES']
            num = 100

        end = datetime.datetime.now()
        print(f'totally time is {end - start}')
        run_time = end - start
        logging(f"run time = {str(run_time)}", args)

        valid_smi = []
        for smi in smiles:
            if valid_smiles(smi):
                mol = Chem.MolFromSmiles(smi)
                Chem.SanitizeMol(mol)
                valid_smi.append(Chem.MolToSmiles(mol))

        save_labeleddata(valid_smi[:110],protein_name,save_dir=csv_dir,num=num)

        # train predictor
        traindataset, testdataset, trainloader, testloader,_ = load_dataloader(data_dir=csv_dir,
                                                                             protein_name=protein_name,args=args)
        logging("data num = %d" % (len(traindataset)), args)
        predictor = DTA1_graph(in_dims=65, emb_dim=args.emb_dim,hidden_dim=args.hidden_dim, smi_hidden_dim=args.smi_dim)
        predictor.to(device)
        optimizer = optim.Adam(predictor.parameters(), lr=args.lr, eps=1e-7)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, last_epoch=-1)

        # wandb.watch(models=predictor)
        train(trainloader=trainloader, testloader=testloader, model=predictor,
              optimizer=optimizer, scheduler=scheduler,criterion=criterion, save_path=save_path,
              device=device, epochs=args.epoch, stop_epoch=args.stop_epoch)

        # test predictor
        testmodel(protein_name, device, args, save_path=save_path)


def main(args):
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    GenAL(device,args)

if __name__ == '__main__':
    args = al_argparser()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    args.log_dir = args.log_dir + str(timestamp) + "-" +args.train_type + "/"
    args.times = str(timestamp)

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
    # init_labeldata(generater_path=args.model_dir, data_path='data/aldata',protein_name=args.protein_name, device=args.device, flag=args.data_flag)
    main(args)