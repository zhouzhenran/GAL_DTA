import pandas as pd
import torch
import argparse
import numpy as np
import random
import os
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import warnings
from dataprocess import valid_smiles
from model import DTA_graph
from score_func import get_train_embedding,get_pool_embedding,compute_firsher_score
from drugex.training.generators import SequenceRNN
from drugex.data.corpus.vocabulary import VocSmiles
from drugex.training.scorers.interfaces import Scorer
from drugex.training.scorers.modifiers import ClippedScore
from drugex.training.environment import DrugExEnvironment
from drugex.training.rewards import ParetoCrowdingDistance,WeightedSum,ParetoTanimotoDistance
from drugex.training.explorers import SequenceExplorer
from drugex.training.monitors import FileMonitor
from drugex.logs import logger

class AffinityScore(Scorer):
    def __init__(self,DTAmodel,dataroot,batchsize,protein_name,smiles_len,invalids_score=0.0, modifier=None):
        super(AffinityScore, self).__init__(modifier)
        self.DTAmodel = DTAmodel
        self.invalidsScore = invalids_score
        self.dataroot = dataroot
        self.batchsize = batchsize
        self.protein_name = protein_name
        self.smiles_len = smiles_len

    def getScores(self,mols,frags=None):
        parsed_mols = []
        if not isinstance(mols[0], str):
            invalids = 0
            for mol in mols:
                parsed_mol = None
                try:
                    parsed_mol = Chem.MolToSmiles(mol) if mol and mol.GetNumAtoms() > 1 else "INVALID"
                    if not valid_smiles(parsed_mol):
                        parsed_mol = "INVALID"
                    if parsed_mol and parsed_mol != "INVALID":
                        pmol = Chem.MolFromSmiles(parsed_mol)
                        Chem.SanitizeMol(pmol)
                        parsed_mol = Chem.MolToSmiles(pmol)
                except Exception as exp:
                    logger.debug(f"Error processing molecule: {parsed_mol} -> \n\t {exp}")
                    parsed_mol = "INVALID"
                if parsed_mol == "INVALID":
                    invalids += 1
                parsed_mols.append(parsed_mol)

            if invalids == len(parsed_mols):
                return np.array([self.invalidsScore] * len(parsed_mols))
        else:
            parsed_mols = mols

        parsed_smiles = []
        scores = np.zeros((len(parsed_mols)))
        for i,smi in enumerate(parsed_mols):
            if smi!= "INVALID":
                parsed_smiles.append(smi)
                scores[i] = -1

        xt = get_pool_embedding(smiles=parsed_smiles,DTAmodel=self.DTAmodel,mol_dir='data/learn',
                                protein_name=self.protein_name,smiles_len=self.smiles_len,)
        x_train = get_train_embedding(DTAmodel=self.DTAmodel, mol_dir=self.dataroot, smiles_len=self.smiles_len, protein_name=self.protein_name)
        trace_scores = compute_firsher_score(xt, x_train, self.batchsize)
        idx = 0
        for i in range(len(parsed_mols)):
            if scores[i] == -1:
                scores[i] = trace_scores[idx]
                idx = idx+1

        return scores

    def getKey(self):
        return f'FisherInfor_{self.protein_name}'

class SimilaritySocre(Scorer):
    def __init__(self,protein_name,fps, invalids_score=0.0, modifier=None):
        super(SimilaritySocre, self).__init__(modifier)
        self.invalidsScore = invalids_score
        self.protein_name = protein_name
        self.fps = fps

    def getScores(self, mols, frags=None):
        parsed_mols = []
        if not isinstance(mols[0], str):
            invalids = 0
            for mol in mols:
                parsed_mol = None
                try:
                    parsed_mol = Chem.MolToSmiles(mol) if mol and mol.GetNumAtoms() > 1 else "INVALID"
                    if parsed_mol and parsed_mol != "INVALID":
                        pmol = Chem.MolFromSmiles(parsed_mol)
                        Chem.SanitizeMol(pmol)
                        parsed_mol = Chem.MolToSmiles(pmol)
                except Exception as exp:
                    logger.debug(f"Error processing molecule: {parsed_mol} -> \n\t {exp}")
                    parsed_mol = "INVALID"
                if parsed_mol == "INVALID":
                    invalids += 1
                parsed_mols.append(parsed_mol)

            if invalids == len(parsed_mols):
                return np.array([self.invalidsScore] * len(parsed_mols))
        else:
            parsed_mols = mols

        parsed_smiles = []
        scores = np.zeros((len(parsed_mols)))
        for i, smi in enumerate(parsed_mols):
            if smi != "INVALID":
                parsed_smiles.append(smi)
                scores[i] = -1

        smi_scores = np.zeros(len(parsed_smiles))
        for i, smi in enumerate(tqdm(parsed_smiles)):
            try:
                mol = AllChem.MolFromSmiles(smi)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                sims = np.array(DataStructs.BulkTanimotoSimilarity(fp,self.fps))
                smi_scores[i] = np.max(sims)
            except:
                continue

        idx = 0
        for i in range(len(parsed_mols)):
            if scores[i] == -1:
                scores[i] = smi_scores[idx]
                idx = idx + 1

        return scores

    def getKey(self):
        return f'Similarity'


class ALFinetune(object):
    def __init__(self,dataroot,model_dir,pre_path,hidden_dims,fea_dim,protein_name,csv_dir,emb_dim,batchsize,restore_epoch,device,epoch,al_iter,smiles_len):
        self.restore_epoch = restore_epoch
        self.dataroot = dataroot
        self.batchsize = batchsize
        self.protein_name = protein_name
        self.emb_dim = emb_dim
        self.hidden_dims = hidden_dims
        self.model_dir = model_dir
        self.pre_path = pre_path
        self.csv_dir = csv_dir
        self.device = torch.device("cuda:" + str(device)) if torch.cuda.is_available() else torch.device("cpu")
        self.gpu = tuple([device])
        self.epoch = epoch
        self.al_iter = al_iter
        self.smiles_len = smiles_len
        self.fea_dim = fea_dim

    def build_model(self):
        self.predictor = DTA_graph(in_dims=65, emb_dim=self.emb_dim,smi_hidden_dim=self.hidden_dims)
        restore_path = os.path.join(self.pre_path, f'G{self.protein_name}_{self.restore_epoch}.pt')
        state_dict = torch.load(restore_path,map_location=lambda storage, loc: storage.cuda(0))
        self.predictor.load_state_dict(state_dict)
        self.predictor.to(self.device)
        self.predictor.eval()

        voc_path = os.path.join(self.model_dir,'chembl_31_smiles_rnn_PT.vocab')
        self.voc = VocSmiles.fromFile(voc_path,encode_frags=False)
        self.finetuned = SequenceRNN(voc=self.voc,is_lstm=True,use_gpus=self.gpu)
        pretained_path = os.path.join(self.model_dir, 'chembl_31_smiles_rnn_PT.pkg')
        train_df = pd.read_csv(os.path.join(self.csv_dir,f'{self.protein_name}_train.csv'))
        train_smiles = train_df['smiles']
        train_mols = [AllChem.MolFromSmiles(smi) for smi in train_smiles]
        fps = [AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024) for mol in train_mols]

        self.AffSocre = AffinityScore(DTAmodel=self.predictor, dataroot=self.csv_dir, smiles_len=self.smiles_len,
                                           batchsize=self.batchsize, protein_name=self.protein_name)
        self.SimSocre = SimilaritySocre(protein_name=self.protein_name,fps=fps)
        self.AffSocre.setModifier(ClippedScore(lower_x=0.2,upper_x=0.8))
        self.SimSocre.setModifier(ClippedScore(lower_x=0.8,upper_x=0.1))
        self.scorers = [
            self.AffSocre,
            self.SimSocre
        ]
        self.thresholds = [
            0.5,
            0.5
        ]

        self.environment = DrugExEnvironment(self.scorers,self.thresholds,reward_scheme=ParetoTanimotoDistance())

    def build_explorer(self):
        self.explorer = SequenceExplorer(
            agent=self.finetuned,
            env=self.environment,
            batch_size=128, # train num
            n_samples=128,# evalute num
            use_gpus=self.gpu
        )
        self.save_dir = os.path.join(self.model_dir,'GAL',self.protein_name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir,exist_ok=True)
        gal_path = os.path.join(self.save_dir,f'{self.protein_name}_{self.al_iter}')
        self.monitor = FileMonitor(gal_path,save_smiles=True,reset_directory=True)

    def finetune(self):
        self.explorer.fit(train_loader=None,monitor=self.monitor,epochs=self.epoch)
