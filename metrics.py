import numpy as np
import torch
from scipy import stats

def get_cindex(gt, pred):
    gt_mask = gt.reshape((1, -1)) > gt.reshape((-1, 1))
    diff = pred.reshape((1, -1)) - pred.reshape((-1, 1))
    h_one = (diff > 0)
    h_half = (diff == 0)
    CI = np.sum(gt_mask * h_one * 1.0 + gt_mask * h_half * 0.5) / np.sum(gt_mask)

    return CI

def cindex_score(y_true,y_pred):
    g = torch.sub(torch.unsqueeze(y_pred,-1),y_pred)
    temp_g = torch.zeros_like(g).float()
    temp_g += torch.where(g==0,torch.tensor(0.5),torch.tensor(0.0))
    temp_g += torch.where(g>0, torch.tensor(1.0),torch.tensor(0.0))

    f = torch.sub(torch.unsqueeze(y_true,-1),y_true) > 0.0
    f = torch.tril(torch.ones_like(f),diagonal=-1) * f.float()

    g = torch.sum(torch.mul(temp_g,f))
    f = torch.sum(f)

    return torch.where(torch.eq(g,0),torch.tensor(0.0),g / f)

def pearson(y,f):
    y = np.array(y).reshape(-1)
    f = np.array(f).reshape(-1)
    rp = np.corrcoef(y, f)
    # print(f'pearson:{rp}')
    return rp[0][1]

def spearman(y,f):
    y = np.array(y).reshape(-1)
    f = np.array(f).reshape(-1)

    rs = stats.spearmanr(y, f)
    # print(f'spearman:{rs}')
    return rs[0]
