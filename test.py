import argparse
import os
import math
from functools import partial

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils

import museval


def batched_predict_liif(model, inp, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred

def batched_predict_mod_sine(model, coord, bsize, latent=None):
    if latent is None:
        pass #TODO
    else:
        with torch.no_grad():
            n = coord.shape[1]
            ql = 0
            preds = []
            while ql<n:
                qr = min(ql + bsize, n)
                mods = model.modulator(latent)
                pred = model.net(coord, mods)
                preds.append(pred)
                ql = qr
            pred = torch.cat(preds,dim =1)
    return pred


def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False):
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    if eval_type is None:
        metric_fn = utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    val_res = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        if eval_bsize is None:
            with torch.no_grad():
                pred = model(latent = latent,coord = batch['coord'])
        else:
            #pred = batched_predict(model, inp,batch['coord'], batch['cell'], eval_bsize)
            pred = batched_predict(model,batch['coord'],latent, eval_bsize)
            

        #pred = pred * gt_div + gt_sub
        #pred.clamp_(0, 1)

        if eval_type is not None: # reshape for shaving-eval
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            pred = pred.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            batch['gt'] = batch['gt'].view(*shape) \
                .permute(0, 3, 1, 2).contiguous()

        res = metric_fn(pred, batch['gt'])
        val_res.add(res.item(), inp.shape[0])

        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))

    return val_res.item()

def eval_sdr(loader, model, verbose=False, 
              latent_list=None, optimizer_config=None, loss_fn=None):
    model.eval()

    val_res = utils.Averager()
    optimize_latent = latent_list is None

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        batch_size = batch['inp'].shape[0]
        inp = batch['inp'].view(1, batch_size, -1, 1)
        if optimize_latent:
            if torch.cuda.device_count() > 1:
                latent = torch.randn(model.module.latent_dim).cuda()
            else:
                latent = torch.randn(model.latent_dim).cuda()
        else:
            latent = latent_list[batch['index'][0]]

        if optimize_latent:
            optimizer = utils.make_optimizer([latent], optimizer_config)
            for i in range(1000):
                optimizer.zero_grad()
                pred = model(latent = latent, coord = batch['coord'])
                loss = loss_fn(pred, batch['gt'])
                loss.backward()
                optimizer.step()
            
        with torch.no_grad():
            pred = model(latent = latent, coord = batch['coord'])
        pred.clamp_(-1, 1)

        sdr = utils.calc_sdr(pred, batch['gt'])
        val_res.add(sdr, inp.shape[0])  # TODO : average for dB

        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))

    return val_res.item()

def eval_sdr_encoder(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False):
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    if eval_type is None:
        metric_fn = museval.evaluate
    else:
        raise NotImplementedError

    val_res = utils.Averager()


    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        batch_size = batch['inp'].shape[0]
        inp = batch['inp'].view(1, batch_size, -1, 1)
        # inp = (batch['inp'] - inp_sub) / inp_div
        with torch.no_grad():
            pred = model(inp = batch['inp'], coord = batch['coord'])
        pred.clamp_(-1, 1)

        try:
            res, _, _, _ = metric_fn(batch['gt'].cpu(), pred.cpu())
        except ValueError as error:
            print("error while calculating sdr: ", error)
            val_res.add(0, inp.shape[0])
            continue

        val_res.add(sum(sum(res)), inp.shape[0])  # TODO

        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))

    return val_res.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        num_workers=8, pin_memory=True)

    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()

    res = eval_psnr(loader, model,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        verbose=True)
    print('result: {:.4f}'.format(res))
