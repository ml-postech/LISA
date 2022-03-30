""" Train for generating LIIF, from image to implicit representation.

    Config:
        train_dataset:
          dataset: $spec; wrapper: $spec; batch_size:
        val_dataset:
          dataset: $spec; wrapper: $spec; batch_size:
        (data_norm):
            inp: {sub: []; div: []}
            gt: {sub: []; div: []}
        (eval_type):
        (eval_bsize):

        model: $spec
        optimizer: $spec
        epoch_max:
        (multi_step_lr):
            milestones: []; gamma: 0.5
        (resume): *.pth

        (epoch_val): ; (epoch_save):
"""

import argparse
import os

import yaml
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR


import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


import datasets
import models
import utils



def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log('{} dataset: size={}'.format(tag, len(dataset)))

    for k, v in dataset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))

    if torch.cuda.device_count() > 1 and tag == 'train':
        sampler = DistributedSampler(dataset, shuffle=(tag == 'train'))
    else:
        sampler = None

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        sampler=sampler, 
        shuffle=(tag == 'train' and sampler == None),
        num_workers=0, pin_memory=True)
        # shuffle=(tag == 'train'), num_workers=2, pin_memory=True)
    return loader, dataset


def make_data_loaders():
    train_loader, train_dataset = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader, val_dataset = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader, train_dataset, val_dataset


def prepare_training(rank, num_gpus):
    device = torch.device(f'cuda:{rank}')

    if config.get('resume') is not None:
        sv_file = torch.load(config['resume'])
        model = models.make(sv_file['model'], load_sd=True)
        if num_gpus > 1:
            model = model.to(device)
            model = DDP(model, device_ids=[rank], output_device=rank)
        else:
            model = model.to(device)
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        for _ in range(epoch_start - 1):
            lr_scheduler.step()

    else:
        model = models.make(config['model'])
        if num_gpus > 1:
            model = model.to(device)
            model = DDP(model, device_ids=[rank], output_device=rank)
        else:
            model = model.to(device)
        optimizer = utils.make_optimizer(
            list(model.parameters()), config['optimizer'])
        epoch_start = 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])

    # model = model.to(device)

    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


def validate(val_loader, model, enc_loss_fn, latent_dim):
    # model.eval()

    val_res = utils.Averager()
    val_samples = []

    pbar = tqdm(val_loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        batch_size = batch['coord'].shape[0]
        setup = {
            'sr': 16000,
            'device': 'cuda',
        }

        z0 = torch.zeros((batch_size, latent_dim)).cuda().requires_grad_(True)
        coord_ = torch.cat([z0.unsqueeze(1).repeat(1, batch['coord'].shape[1], 1), batch['coord'][:, :, 1:2]], dim=2)
        pred_ = model(coord_)
        loss1, _ = enc_loss_fn(pred_, batch['gt'], setup=setup)
        grad = torch.autograd.grad(loss1, [z0], create_graph=True, retain_graph=True)[0]
        # z1 = -grad * batch_size
        latents = -grad * batch_size
    
        # # 2-step GON
        # coord_ = torch.cat([z1.unsqueeze(1).repeat(1, batch['coord'].shape[1], 1), batch['coord'][:, :, 1:2]], dim=2)
        # pred_ = model(coord_)
        # loss2, _ = enc_loss_fn(pred_, batch['gt'], setup=setup)
        # grad = torch.autograd.grad(loss2, [z0], create_graph=True, retain_graph=True)[0]
        # latents = -grad * batch_size + z1
    
        coord = torch.cat([latents.unsqueeze(1).repeat(1, batch['coord'].shape[1], 1), batch['coord'][:, :, 1:2]], dim=2)
        pred = model(coord)
        pred.clamp_(-1, 1)

        sdr = utils.calc_sdr(pred, batch['gt'])
        val_res.add(sdr, batch_size)  # TODO : average for dB
        val_samples.append(pred.view(-1))

    return val_res.item(), (torch.cat(val_samples, dim=0).detach().cpu(), setup['sr'])


def train(train_loader, model, optimizer, loss_fn, enc_loss_fn, latent_dim):
    model.train()
    train_loss = utils.Averager()
    loss_logging = {}
    train_samples = None

    for batch in tqdm(train_loader, leave=False, desc='train'):
        for k, v in batch.items():
            batch[k] = v.cuda()  

        optimizer.zero_grad()

        batch_size = batch['coord'].shape[0]
        setup = {
            'sr': 16000,
            'device': 'cuda',
        }

        z0 = torch.zeros((batch_size, latent_dim)).cuda().requires_grad_(True)
        coord_ = torch.cat([z0.unsqueeze(1).repeat(1, batch['coord'].shape[1], 1), batch['coord'][:, :, 1:2]], dim=2)
        pred_ = model(coord_)
        loss1, _ = enc_loss_fn(pred_, batch['gt'], setup=setup)
        grad = torch.autograd.grad(loss1, [z0], create_graph=True, retain_graph=True)[0]
        # z1 = -grad * batch_size
        latents = -grad * batch_size
    
        # # 2-step GON
        # coord_ = torch.cat([z1.unsqueeze(1).repeat(1, batch['coord'].shape[1], 1), batch['coord'][:, :, 1:2]], dim=2)
        # pred_ = model(coord_)
        # loss2, _ = enc_loss_fn(pred_, batch['gt'], setup=setup)
        # grad = torch.autograd.grad(loss2, [z0], create_graph=True, retain_graph=True)[0]
        # latents = -grad * batch_size + z1
    
        coord = torch.cat([latents.unsqueeze(1).repeat(1, batch['coord'].shape[1], 1), batch['coord'][:, :, 1:2]], dim=2)
        pred = model(coord)

        loss, loss_logs = loss_fn(pred, batch['gt'], setup=setup)

        if torch.cuda.device_count() > 1:
            dist.all_reduce(loss)
            num_gpu = torch.cuda.device_count()
            loss = loss / num_gpu
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    

        for k, v in loss_logs.items():
            if not (k in loss_logging.keys()):
                loss_logging[k] = utils.Averager()
            loss_logging[k].add(v)

        if train_samples is None:
            sample_chunk = pred.detach().clone()
            train_samples = sample_chunk
            sr = setup['sr']

        train_loss.add(loss.item())


    return train_loss.item(), \
        {k: v.item() for k, v in loss_logging.items()}, \
        (train_samples.view(-1).cpu(), sr)


def main(config_, save_path, rank=0, num_gpus=1):
    global config, log, writer
    config = config_ # (1) config를 불러오고
    log, writer = utils.set_save_path(save_path, remove=False, rank=rank)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    random_seed = 0
    utils.random_seed(random_seed)

    train_loader, val_loader, train_dataset, _ = make_data_loaders()

    model, optimizer, epoch_start, lr_scheduler = prepare_training(rank, num_gpus)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        loss_fn = utils.get_loss_fn(config['loss'])
        enc_loss_fn = utils.get_loss_fn(config['enc_loss'])
        latent_dim = config['latent_dim']
        
        train_loss, train_losses, train_sample = train(train_loader, model, optimizer, loss_fn, enc_loss_fn, latent_dim) 
        if lr_scheduler is not None:
            lr_scheduler.step()

        log_info.append('train: loss={:.4f}'.format(train_loss))
        if torch.cuda.device_count() > 1:
            train_loss_ = torch.Tensor([train_loss]).cuda()
            dist.reduce(train_loss_, 0)
            train_loss = train_loss_.cpu().item() / float(dist.get_world_size())
        if writer is not None:
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('loss/train', train_loss, epoch)
            for k, v in train_losses.items():
                writer.add_scalar(f'loss/{k}', v, epoch)
            if epoch % config['epoch_val'] == 0:
                writer.add_audio(tag="sample/train", snd_tensor=train_sample[0].view(-1), global_step=epoch, sample_rate=train_sample[1]) #TODO: sr


        if num_gpus > 1:
            model_ = model.module
        else:
            model_ = model

        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()

        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }

        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,
                os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            if num_gpus > 1 and (config.get('eval_bsize') is not None):
                model_ = model
            else:
                model_ = model
            
            val_res, val_sample = validate(val_loader, model_, enc_loss_fn, latent_dim)


            log_info.append('val: sdr={:.4f}'.format(val_res))
            if torch.cuda.device_count() > 1:
                val_res_ = torch.Tensor([val_res]).cuda()
                dist.reduce(val_res_, 0)
                val_res = val_res_.cpu().item() / float(dist.get_world_size())
            if writer is not None:
                writer.add_scalar('sdr', val_res, epoch)
                writer.add_audio(tag="sample/val", snd_tensor=val_sample[0].view(-1), global_step=epoch, sample_rate=val_sample[1])
            if val_res > max_val_v:
                max_val_v = val_res
                torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        if writer is not None:
            writer.flush()

    if writer is not None:
        writer.add_hparams(
            hparam_dict={
                'model': config['model']['name'],
                    'w0_initial': config['model']['args']['w0_initial'],
                    'w0': config['model']['args']['w0'],
                'loss': config['loss'],
                'lr': config['optimizer']['args']['lr'],
            },
            metric_dict={
                'loss': train_loss,
                'sdr': max_val_v
            }
        )
        writer.flush()

def setup_process(rank, num_gpus, config, save_path):
    dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=num_gpus)
    torch.cuda.set_device(rank)
    main(config, save_path, rank=rank, num_gpus=num_gpus)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--save_path', default='./save')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join(args.save_path, save_name)
    config['name'] = save_name

    if torch.cuda.device_count() > 1:
        num_gpus = torch.cuda.device_count()
        mp.spawn(setup_process, args=(num_gpus, config, save_path,), nprocs=num_gpus, join=True)
    else:
        main(config, save_path)
