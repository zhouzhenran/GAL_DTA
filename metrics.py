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

def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult
    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))
    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def get_rm2(ys_orig, ys_line):
    # ys_line = np.concatenate(ys_line)
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))

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
    print(f'pearson:{rp}')
    return rp[0]
def spearman(y,f):
    y = np.array(y).reshape(-1)
    f = np.array(f).reshape(-1)

    rs = stats.spearmanr(y, f)
    print(f'spearman:{rs}')
    return rs[0]
