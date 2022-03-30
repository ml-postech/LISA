import os
import time
import shutil
import math
import random

import torch
import torch.nn as nn
import numpy as np
import museval
import auraloss
from torch.optim import SGD, Adam, AdamW
from tensorboardX import SummaryWriter
import librosa
import torchaudio.transforms as T
import loss


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_save_path(save_path, remove=True, rank=0):
    if rank == 0:
        ensure_path(save_path, remove=remove)
        set_log_path(save_path) 
        writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    else:
        save_path = os.path.join(save_path, str(rank))
        ensure_path(save_path, remove=remove)
        set_log_path(save_path) 
        writer = None
    return log, writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam,
        'adamW': AdamW,
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    coord = make_coord(img.shape[-2:])
    rgb = img.view(3, -1).permute(1, 0)
    return coord, rgb


def calc_psnr(pred, gt):
    # require [nsrc, nsample, nchan]
    eps = 1e-8
    l2_loss = (gt-pred).norm(2, dim=[1,2])
    # l2_norm = gt.norm(2, dim=[1,2])
    snr = -20 * torch.log10(l2_loss + eps)
    return torch.mean(snr)

def calc_sdr(pred, gt):
    try:
        # require [nsrc, nsample, nchan]
        res, _, _, _ = museval.evaluate(gt.view(1, -1, 1).detach().cpu().numpy(), pred.view(1, -1, 1).detach().cpu().numpy())
    except ValueError as error:
        print("error while calculating sdr: ", error)
        return np.nan
    return np.average(res)

# https://en.wikipedia.org/wiki/Signal-to-noise_ratio
# https://github.com/kuleshov/audio-super-res/blob/master/src/models/model.py
def calc_snr(pred, gt):
    # require [nsrc, nsample, nchan]
    eps = 1e-8
    l2_loss = (gt-pred).norm(2, dim=[1,2])
    l2_norm = gt.norm(2, dim=[1,2])
    snr = 20 * torch.log(l2_norm / l2_loss + eps) / math.log(10)
    return torch.mean(snr)


def get_power(x):
    S = T.Spectrogram(n_fft=2048)(x)
    S = torch.log(torch.abs(S)**2 + 1e-8) / math.log(10)
    return S

def compute_log_distortion(x_pr, x_hr):
    x_hr = torch.flatten(x_hr)
    x_pr = torch.flatten(x_pr)
    S1 = get_power(x_hr)
    S2 = get_power(x_pr)
    lsd = torch.mean(torch.sqrt(torch.mean((S1-S2)**2 + 1e-8, dim=1)), dim=0)
    return min(lsd, 10.)

def get_loss_fn(name):
    
    if name == 'l2':
        loss_l2 = nn.MSELoss()
        def loss_fn(pred, target, setup=None):
            loss_l2_ = loss_l2(pred, target)
            return loss_l2_, \
                {
                    'l2_loss': loss_l2_.detach().clone(),
                }
    elif name == 'l1':
        loss_l1 = nn.L1Loss()
        def loss_fn(pred, target, setup=None):
            loss_l1_ = loss_l1(pred, target)
            return loss_l1_, \
                {
                    'l1_loss': loss_l1_.detach().clone(),
                }
    elif name == 'l2_spec':
        loss_l2 = nn.MSELoss()
        loss_spec = auraloss.freq.STFTLoss()
        # loss_spec input should be [B, C, t]
        spec_coeff = 0.001
        def loss_fn(pred, target, setup=None):
            loss_l2_ = loss_l2(pred, target)
            loss_spec_ = spec_coeff * loss_spec(pred.permute(0, 2, 1), target.permute(0, 2, 1))
            return loss_l2_ + loss_spec_, \
                {
                    'l2_loss': loss_l2_,
                    'spec_loss': loss_spec_,
                }   
    elif name == 'l1_spec':
        loss_l1 = nn.L1Loss()
        loss_spec = auraloss.freq.STFTLoss()
        # loss_spec input should be [B, C, t]
        spec_coeff = 0.001
        def loss_fn(pred, target, setup=None):
            loss_l1_ = loss_l1(pred, target)
            loss_spec_ = spec_coeff * loss_spec(pred.permute(0, 2, 1), target.permute(0, 2, 1))
            return loss_l1_ + loss_spec_, \
                {
                    'l1_loss': loss_l1_,
                    'spec_loss': loss_spec_,
                }   
    elif name == 'l1_mel':
        loss_l1 = nn.L1Loss()
        # loss_spec input should be [B, C, t]
        spec_coeff = 0.001
        def loss_fn(pred, target, setup=None):
            loss_spec = auraloss.freq.MelSTFTLoss(setup['sr'], device=setup['device'])

            loss_l1_ = loss_l1(pred, target)
            loss_spec_ = spec_coeff * loss_spec(pred.permute(0, 2, 1), target.permute(0, 2, 1))
            return loss_l1_ + loss_spec_, \
                {
                    'l1_loss': loss_l1_,
                    'spec_loss': loss_spec_,
                }
    elif name == 'l1_multi_spec':
        loss_l1 = nn.L1Loss()
        loss_spec = auraloss.freq.MultiResolutionSTFTLoss()
        # loss_spec input should be [B, C, t]
        spec_coeff = 0.001
        def loss_fn(pred, target, setup=None):
            loss_l1_ = loss_l1(pred, target)
            loss_spec_ = spec_coeff * loss_spec(pred.permute(0, 2, 1), target.permute(0, 2, 1))
            return loss_l1_ + loss_spec_, \
                {
                    'l1_loss': loss_l1_.detach().clone(),
                    'spec_loss': loss_spec_.detach().clone(),
                }
    elif name == 'gan_loss':
        loss_fn = loss.GANLoss()
    else:
        loss_fn = nn.L1Loss()

    return loss_fn

def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    