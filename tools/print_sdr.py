import argparse
import os

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

import numpy as np
import museval


def calc_sdr(pred, gt):
    try:
        # require [nsrc, nsample, nchan]
        res, _, _, _ = museval.evaluate(gt.view(1, -1, 1).detach().cpu().numpy(), pred.view(1, -1, 1).detach().cpu().numpy())
    except ValueError as error:
        print("error while calculating sdr: ", error)
        return np.nan
    return np.average(res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', default='original.wav')
    parser.add_argument('--predict', default='pred.wav')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    gt, sr1 = torchaudio.load(args.original)
    pred, sr2 = torchaudio.load(args.predict)
    
    gt = gt.view(-1, 1)
    # gt = gt[:, ::4].view(-1, 1)
    pred = pred.view(-1, 1)
    
    print(calc_sdr(pred, gt))
