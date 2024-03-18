import numpy as np
import json
import pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import AllChem
from pubchemfp import GetPubChemFPs

# protein
CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21,
               "V": 22, "Y": 23, "X": 24,
               "Z": 25}

CHARPROTLEN = 25

# smiles
CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64

def valid_smiles(smiles):
    for c in smiles:
        if c not in CHARISOSMISET:
            return False
    return True

def label_smiles(line, MAX_SMI_LEN, smi_ch_ind):
    X = np.zeros(MAX_SMI_LEN)
    for i, ch in enumerate(line[:MAX_SMI_LEN]):  # x, smi_ch_ind, y
        X[i] = smi_ch_ind[ch]
    return X  # .tolist()

def label_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
    X = np.zeros(MAX_SEQ_LEN)

    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]

    return X  # .tolist()

def fp_smiles(smi,fp_type='mixed'):
    mol = Chem.MolFromSmiles(smi)
    fp = []
    if fp_type == 'mixed':
        fp1 = AllChem.GetMACCSKeysFingerprint(mol)  # 167
        fp2 = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1)  # 441
        fp3 = GetPubChemFPs(mol)
        # fp3 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)  # 1024
        fp.extend(fp1)
        fp.extend(fp2)
        fp.extend(fp3)
    else:
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fp.extend(fp1)
    return fp


def SW_parser(line, PROTEIN_COUNT):
    X = np.zeros(PROTEIN_COUNT)
    lineNumstr = line.split("|")

    for i in range(0, PROTEIN_COUNT):
        X[i] = float(lineNumstr[i])

    return X

# dataset
class DataSet(object):
    def __init__(self, f_path, setting, smilen):
        self.FPATH = f_path

        self.SMILEN = smilen
        self.charseqset = CHARPROTSET
        self.charseqset_size = CHARPROTLEN

        self.charsmiset = CHARISOSMISET
        self.charsmiset_size = CHARISOSMILEN
        self.PROBLEMSET = setting

    def read_sets(self):
        fpath = self.FPATH
        setting_no = self.PROBLEMSET # 1-正常 2,3,4-cold start
        print("Reading %s start" % fpath)

        test_fold = json.load(open(fpath + "folds/test_fold_setting" + str(setting_no) + ".txt"))
        train_folds = json.load(open(fpath + "folds/train_fold_setting" + str(setting_no) + ".txt"))

        return test_fold, train_folds

    def parse_data(self,protype=1):
        fpath = self.FPATH
        print("Read %s start" % fpath)

        ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
        if protype==1:
            proteins = json.load(open(fpath + "protVeclzma1.txt"), object_pairs_hook=OrderedDict)
            proteins1 = json.load(open(fpath + "protVecSW.txt"), object_pairs_hook=OrderedDict)
        else:
            proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)

        Y = pickle.load(open(fpath + "Y", "rb"), encoding='latin1')  ### TODO: read from raw

        XD = []
        XT = []
        SMILES = []
        FPS = []
        if protype == 1:
            proteinFeatures = int(proteins["num"])
            ckey = 0
            for t in proteins.keys():
                if ckey >= 1:
                    if fpath == 'data/davis/':
                        XT.append((SW_parser(proteins[t], proteinFeatures)) * (
                                SW_parser(proteins1[t], proteinFeatures) / 100))
                    else:
                        XT.append(
                            (SW_parser(proteins[t], proteinFeatures)) * (SW_parser(proteins1[t], proteinFeatures)))

                ckey = ckey + 1
        else:
            for t in proteins.keys():
                XT.append(proteins[t])

        for d in ligands.keys():
            XD.append(label_smiles(ligands[d], self.SMILEN, self.charsmiset))
            SMILES.append(ligands[d])
            FPS.append(fp_smiles(ligands[d]))

        return XD, XT, Y, SMILES, FPS