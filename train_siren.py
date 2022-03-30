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

    if torch.cuda.device_count() > 1:
        sampler = DistributedSampler(dataset, shuffle=(tag == 'train'))
    else:
        sampler = None

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        sampler=sampler,
        num_workers=1, pin_memory=True)
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
        latent_list = sv_file['latent_list']

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
        latent_list = None

    # model = model.to(device)

    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler, latent_list


def validate(val_loader, model):
    model.eval()

    val_res = utils.Averager()

    val_samples = []

    pbar = tqdm(val_loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        batch_size = batch['inp'].shape[0]
        with torch.no_grad():
            pred = model(coord = batch['coord'][:, :, 1:2])
        pred.clamp_(-1, 1)

        sdr = utils.calc_sdr(pred, batch['gt'])
        val_res.add(sdr, batch_size)  # TODO : average for dB
        val_samples.append(pred.view(-1))

    return val_res.item(), (torch.cat(val_samples, dim=0).detach().cpu(), 48000)


def train(train_loader, model, optimizer, loss_fn):
    model.train()
    train_loss = utils.Averager()
    loss_logging = {}
    train_samples = []

    if hasattr(train_loader.dataset, 'reinit_down_sr'):
        train_loader.dataset.reinit_down_sr()

    for batch in tqdm(train_loader, leave=False, desc='train'):
        for k, v in batch.items():
            batch[k] = v.cuda()  

        
        optimizer.zero_grad()

        batch_size = batch['coord'].shape[0]
        loss = 0
        loss_logging_batch = {}
        setup = {
            'sr': 48000,
            'device': 'cuda',
        }

        for i in range(batch_size):
            pred = model(coord=batch['coord'][i:i+1, :, 1:2])

            gt = batch['gt'][i:i+1]

            if 'offset' in batch:
                for j in range(batch['offset'].shape[1]):
                    offset = batch['offset'][i, j].item()
                    length = int(batch['len'][i, j].item())
                    sample_chunk = pred.view(1, -1, 1)[:, offset:offset+length, :]
                    gt_ = gt[:, offset:offset+length, :]

                    sr = int(batch['coord'][i, offset, 0].item() * 48000 / train_loader.dataset.sr_scale)
                    setup['sr'] = sr

                    loss_t, loss_logs = loss_fn(sample_chunk, gt_, setup)
                    loss += loss_t
                    for k, v in loss_logs.items():
                        if not (k in loss_logging_batch.keys()):
                            loss_logging_batch[k] = 0
                        loss_logging_batch[k] += v.item()

                length = int(batch['len'][i, 0].item())
                sr = int(batch['coord'][i, 0, 0].item() * 48000 / train_loader.dataset.sr_scale)
                sample_chunk = pred.view(-1)[0:length].detach().clone().cpu()
                train_samples.append(sample_chunk)
            else:
                loss_t, loss_logs = loss_fn(pred, gt, setup)
                loss += loss_t
                for k, v in loss_logs.items():
                    if not (k in loss_logging_batch.keys()):
                        loss_logging_batch[k] = 0
                    loss_logging_batch[k] += v.item()

                sample_chunk = pred.detach().clone().cpu()
                train_samples.append(sample_chunk)
                sr = 48000

        loss = loss / batch_size
        loss_logging_batch = {k: v / batch_size for k, v in loss_logging_batch.items()}
        if 'offset' in batch:
            loss = loss / batch['offset'].shape[1]
            loss_logging_batch = {k: v / batch['offset'].shape[1] for k, v in loss_logging_batch.items()}


        train_loss.add(loss.item())
        for k, v in loss_logging_batch.items():
            if not (k in loss_logging.keys()):
                loss_logging[k] = utils.Averager()
            loss_logging[k].add(v)

        loss.backward()
        optimizer.step()

    return train_loss.item(), \
        {k: v.item() for k, v in loss_logging.items()}, \
        (torch.cat(train_samples), sr)


def main(config_, save_path, rank=0, num_gpus=1):
    global config, log, writer
    config = config_ # (1) config를 불러오고
    log, writer = utils.set_save_path(save_path, remove=False, rank=rank)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    random_seed = 0
    utils.random_seed(random_seed)

    train_loader, val_loader, train_dataset, _ = make_data_loaders()

    model, optimizer, epoch_start, lr_scheduler, _ = prepare_training(rank, num_gpus)


    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        loss_fn = utils.get_loss_fn(config['loss'])
        
        train_loss, train_losses, train_sample = train(train_loader, model, optimizer, loss_fn) 
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
            'epoch': epoch,
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
            
            val_res, val_sample = validate(val_loader, model_)

            
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
                'loss': config['loss'],
                'lr': config['optimizer']['args']['lr'],
            },
            metric_dict={
                'loss': train_loss,
                'sdr': val_res
            }
        )
        writer.flush()

def setup_process(rank, num_gpus, config, save_path):
    dist.init_process_group("nccl", rank=rank, world_size=num_gpus)
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
