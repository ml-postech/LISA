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
from scipy import interpolate
import matplotlib.pyplot as plt


import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


import datasets
import models
import utils



def make_data_loader(spec, target_sr=None, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    if target_sr is None:
        dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    else:
        dataset = datasets.make(spec['wrapper'], args={'dataset': dataset, 'gt_sr': target_sr})

    print('{} dataset: size={}'.format(tag, len(dataset)))

    for k, v in dataset[0].items():
        print('  {}: shape={}'.format(k, tuple(v.shape)))

    if torch.cuda.device_count() > 1 and tag == 'train':
        sampler = DistributedSampler(dataset, shuffle=(tag == 'train'))
    else:
        sampler = None

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        sampler=sampler, 
        shuffle=(tag == 'train' and sampler == None),
        num_workers=4, pin_memory=True)
        # shuffle=(tag == 'train'), num_workers=2, pin_memory=True)
    return loader, dataset


def make_data_loaders(target_sr):
    val_loader, val_dataset = make_data_loader(config.get('val_dataset'), tag='val', target_sr=target_sr)
    return val_loader, val_dataset


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

    print('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler

def plot_waveform(waveform, sample_rate, filename, title="Waveform", xlim=None, ylim=None):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].plot(time_axis, waveform[c], linewidth=1)
    axes[c].grid(True)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
    if ylim:
      axes[c].set_ylim(ylim)
  figure.suptitle(title)
  plt.show(block=False)
  plt.savefig(filename)


def plot_specgram(waveform, sample_rate, filename, title="Spectrogram", xlim=None):
  waveform = waveform.numpy()

  font = {
    'family' : 'serif',
    'size'   : 18,
    # 'usetex' : True,
    }
  plt.rc('font', **font)

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].specgram(waveform[c], Fs=sample_rate)
    axes[c].set_yticks([5000, 10000, 15000, 20000])
    axes[c].set_yticklabels(["5k", "10k", "15k", "20k"])
    # if num_channels > 1:
    #   axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
#   figure.suptitle(title)
#   plt.show(block=False)
  plt.savefig(filename)
  plt.close()



def validate(val_loader, model, target_sr=None):
    # model.eval()

    val_res = utils.Averager()
    val_res_snr = utils.Averager()
    val_ref_snr = utils.Averager()
    val_res_lsd = utils.Averager()
    val_ref_lsd = utils.Averager()
    val_samples = []

    ii = 0
    pbar = tqdm(val_loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        batch_size = batch['coord'].shape[0]
        setup = {
            'sr': target_sr,
            'device': 'cuda',
        }

        pred = model(batch)
        pred.clamp_(-1, 1)

        index = 0

        if ii == 3:
            with torch.no_grad():
                # sdr = utils.calc_sdr(pred, batch['gt'])
                snr = utils.calc_snr(pred.detach().clone(), batch['gt'])
                lsd = utils.compute_log_distortion(pred.detach().clone().cpu(), batch['gt'].cpu()) ######
                
                # from torchaudio import transforms
                # ref_up = transforms.Resample(8000, target_sr, resampling_method='sinc_interpolation').cuda()(batch['inp'].permute(0,2,1)).permute(0,2,1)
                # lr_x = np.linspace(1, 8000, 8000)
                # hr_x = np.linspace(1, 8000, 27000)
                # f = interpolate.splrep(lr_x, batch['inp'][index,:0].detach().cpu().numpy())
                # ref_up = interpolate.splev(hr_x, f)
                # ref_up = torch.Tensor(ref_up).view(1, -1, 1)
                # ref_snr = utils.calc_snr(ref_up, batch['gt'][index,:,0].cpu())
                # ref_lsd = utils.compute_log_distortion(ref_up.cpu(), batch['gt'][index,:,0].cpu()) ######

                # ref_snr = utils.calc_snr(ref_up, batch['gt'])
                # ref_lsd = utils.compute_log_distortion(ref_up.cpu(), batch['gt'].cpu()) ######
            # val_res.add(sdr, batch_size)  # TODO : average for dB
            val_res_snr.add(snr, batch_size)  # TODO : average for dB
            val_res_lsd.add(lsd, batch_size)
            # val_ref_snr.add(ref_snr, 1)  # TODO : average for dB
            # val_ref_lsd.add(ref_lsd, 1)

            # if ii == 0:            
                # plot_waveform((pred - batch['gt'])[0, :, 0].unsqueeze(0).detach().cpu(), sample_rate=8000, filename="diff_autoenc_12.png")
            # val_samples.append(pred.view(-1).detach().cpu())
            
            # if ii == 3:
            p_gt = batch['gt'][index,:,0].cpu().unsqueeze(0)
            p_pred = pred[index, :, 0].detach().cpu().unsqueeze(0)
            plot_specgram(p_gt, 48000, f"gt{ii}_{index}.pdf")
            # # plot_specgram(ref_up[index, :, 0].unsqueeze(0), 48000, "spline")
            plot_specgram(p_pred, 48000, f"pred{ii}_{index}.pdf")
            # plot_specgram(p_gt - p_pred, 48000, f"images/diff{ii}_{index}")

        ii += 1

    return (val_res.item(), val_res_snr.item(), val_ref_snr.item(), val_ref_lsd.item()), \
        None, val_res_lsd.item()


def main(config_, save_path, target_sr=None, rank=0, num_gpus=1):
    global config, log
    config = config_ # (1) config를 불러오고

    random_seed = 0
    utils.random_seed(random_seed)

    val_loader, _ = make_data_loaders(target_sr)

    model, _, epoch_start, lr_scheduler = prepare_training(rank, num_gpus)
    model.local_ensemble = True

    timer = utils.Timer()

    epoch = epoch_start
    t_epoch_start = timer.t()

    if num_gpus > 1:
        model_ = model.module
    else:
        model_ = model


    if num_gpus > 1 and (config.get('eval_bsize') is not None):
        model_ = model
    else:
        model_ = model
    
    val_res, val_sample, val_lsd = validate(val_loader, model_, target_sr=target_sr)

    # print('val: sdr={:.4f}'.format(val_res[0]))
    print('eval: snr={:.4f}'.format(val_res[1]))
    print('eval: lsd = {:.4f}'.format(val_lsd))
    print('eval: ref_snr={:.4f}'.format(val_res[2]))
    print('eval: ref_lsd={:.4f}'.format(val_res[3]))

    t = timer.t()
    t_epoch = utils.time_text(t - t_epoch_start)
    print('{}'.format(t_epoch))



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
    parser.add_argument('--sr', default=None)
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

    main(config, save_path, target_sr=args.sr)
