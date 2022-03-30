import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import audio_dataset, register
from utils import to_pixel_samples, make_coord


from torchaudio import transforms as audioTransform



@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }


def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(img)))


@register('sr-implicit-downsampled')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            img = img[:, :round(h_lr * s), :round(w_lr * s)] # assume round int
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            w_lr = self.inp_size
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }


@register('sr-implicit-uniform-varied')
class SRImplicitUniformVaried(Dataset):

    def __init__(self, dataset, size_min, size_max=None,
                 augment=False, gt_resize=None, sample_q=None):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]
        p = idx / (len(self.dataset) - 1)
        w_hr = round(self.size_min + (self.size_max - self.size_min) * p)
        img_hr = resize_fn(img_hr, w_hr)

        if self.augment:
            if random.random() < 0.5:
                img_lr = img_lr.flip(-1)
                img_hr = img_hr.flip(-1)

        if self.gt_resize is not None:
            img_hr = resize_fn(img_hr, self.gt_resize)

        hr_coord, hr_rgb = to_pixel_samples(img_hr)

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / img_hr.shape[-2]
        cell[:, 1] *= 2 / img_hr.shape[-1]

        return {
            'inp': img_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }


def resize_audio_fn(audio, sr, down_sr):
    resampler = audioTransform.Resample(sr, down_sr, resampling_method='sinc_interpolation')
    return resampler(audio)


@register('sr-implicit-audio-downsampled')
class SRImplicitAudioDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None, coord_scale=1, sr_scale=1):
        self.dataset = dataset
        # self.inp_size = inp_size
        # self.scale_min = scale_min
        # if scale_max is None:
        #     scale_max = scale_min
        # self.scale_max = scale_max
        # self.augment = augment
        # self.sample_q = sample_q
        self.coord_scale = coord_scale
        self.sr_scale = sr_scale
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        #audio, sr, _, _, _ = self.dataset[idx]
        audio, sr = self.dataset[idx]
        audio = torch.mean(audio[:, :], dim=0).view(1, -1)

        down_sr = 24000
        down_audio = resize_audio_fn(audio, sr, down_sr)

        coord = make_coord(audio.shape, ranges=[(self.sr_scale * down_sr / 48000, self.sr_scale * down_sr / 48000), (-self.coord_scale, self.coord_scale)])

        #hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        # if self.sample_q is not None:
        #     #sample_lst = np.random.choice(
        #         len(hr_coord), self.sample_q, replace=False)
        #     hr_coord = hr_coord[sample_lst]
        #     hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / audio.shape[-1]
        

        return {
            'inp': down_audio.view(-1, 1),
            'coord': coord,
            'cell': cell,
            'gt': audio.view(-1, 1),
        }


@register('audio-chunked-liif')
class AudioLIIFCoordChunked(Dataset):
    def __init__(self, dataset, chunk_len=1, coord_scale=1., sr_scale=1.,
            local_chunk_len=48, sample_q=None, gt_aug_max=None,
            gt_sr=24000, input_sr=24000):
        self.dataset = dataset
        self.chunk_len = chunk_len
        self.local_chunk_len = local_chunk_len
        self.coord_scale = coord_scale
        self.sr_scale = sr_scale
        self.gt_sr = gt_sr
        self.input_sr = input_sr
        self.sample_q = sample_q
        self.gt_aug_max = gt_aug_max

        self.cached_data = {}
        self.init_idx_mapping()
        
    def init_idx_mapping(self):
        self.idx_mapping = []
        for i in range(len(self.dataset)):
            wave, sr = self.dataset[i]
            wave_len = wave.shape[1]
            chunk_wave_len = int(self.chunk_len * sr)
            chunk_num = int(wave_len // chunk_wave_len) + 1

            for j in range(chunk_num):
                chunk_offset = int((wave_len - chunk_wave_len) * (j / (chunk_num - 1)))
                self.idx_mapping.append((i, chunk_offset, chunk_wave_len))
            assert self.idx_mapping[-1][1] + chunk_wave_len == wave_len

    def __len__(self):
        return len(self.idx_mapping)

    def __getitem__(self, idx):
        if idx in self.cached_data.keys() and self.gt_aug_max is None:
            return self.cached_data[idx]
        dataset_idx, offset, chunk_len = self.idx_mapping[idx]
        #audio, sr, _, _, _ = self.dataset[idx]
        full_audio, sr = self.dataset[dataset_idx]
        audio = torch.mean(full_audio[:, offset:offset+chunk_len], dim=0).view(1, -1)
        
        if self.gt_aug_max is not None:
            min_unit = 100
            gt_sr = random.randrange(self.input_sr, int(self.input_sr * self.gt_aug_max))
            gt_sr = int(gt_sr / min_unit) * min_unit
        else:
            gt_sr = self.gt_sr
        gt_audio = resize_audio_fn(audio, sr, gt_sr)
        gt_coord = make_coord(gt_audio.shape, ranges=[(-1, 1), 
                                                    (-self.coord_scale, self.coord_scale)])
        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(gt_coord), self.sample_q, replace=False)
            gt_coord = gt_coord[sample_lst, :]
            gt_audio = gt_audio.view(-1, 1)
            gt_audio = gt_audio[sample_lst, :]

        down_audio = resize_audio_fn(audio, sr, self.input_sr)
        down_coord = make_coord(down_audio.shape, ranges=[(-1, 1), 
                                                            (-self.coord_scale, self.coord_scale)])

        data = {
            'inp_coord': down_coord,
            'inp': down_audio.view(-1, 1),
            'coord': gt_coord,
            'gt': gt_audio.view(-1, 1),
            'index': torch.LongTensor([idx])
        }
        self.cached_data[idx] = data
        return data


@register('audio-chunked')
class AudioCoordChunked(Dataset):
    def __init__(self, dataset, chunk_len=0.2, coord_scale=1., sr_scale=1.,
            gt_sr=24000, input_sr=24000):
        self.dataset = dataset
        self.chunk_len = chunk_len
        self.coord_scale = coord_scale
        self.sr_scale = sr_scale
        self.gt_sr = gt_sr
        self.input_sr = input_sr

        self.cached_data = {}
        self.init_idx_mapping()
        
    def init_idx_mapping(self):
        self.idx_mapping = []
        for i in range(len(self.dataset)):
            wave, sr = self.dataset[i]
            wave_len = wave.shape[1]
            chunk_wave_len = int(self.chunk_len * sr)
            chunk_num = int(wave_len // chunk_wave_len) + 1

            for j in range(chunk_num):
                chunk_offset = int((wave_len - chunk_wave_len) * (j / (chunk_num - 1)))
                self.idx_mapping.append((i, chunk_offset, chunk_wave_len))
            assert self.idx_mapping[-1][1] + chunk_wave_len == wave_len

    def __len__(self):
        return len(self.idx_mapping)

    def __getitem__(self, idx):
        if idx in self.cached_data.keys():
            return self.cached_data[idx]
        dataset_idx, offset, chunk_len = self.idx_mapping[idx]
        #audio, sr, _, _, _ = self.dataset[idx]
        full_audio, sr = self.dataset[dataset_idx]
        audio = torch.mean(full_audio[:, offset:offset+chunk_len], dim=0).view(1, -1)
        
        gt_audio = resize_audio_fn(audio, sr, self.gt_sr)
        gt_coord = make_coord(gt_audio.shape, ranges=[(self.sr_scale * self.gt_sr / 48000, self.sr_scale * self.gt_sr / 48000), (-self.coord_scale, self.coord_scale)])

        down_audio = resize_audio_fn(audio, sr, self.input_sr)
        down_coord = make_coord(down_audio.shape, ranges=[(self.sr_scale * self.input_sr / 48000, self.sr_scale * self.input_sr / 48000), (-self.coord_scale, self.coord_scale)])

        data = {
            'inp_coord': down_coord,
            'inp': down_audio.view(-1, 1),
            'coord': gt_coord,
            'gt': gt_audio.view(-1, 1),
            'index': torch.LongTensor([idx])
        }
        self.cached_data[idx] = data
        return data


@register('audio-chunked-down')
class AudioCoordChunkedDownsampled(AudioCoordChunked):
    def __init__(self, dataset, chunk_len=0.2, coord_scale=1., sr_scale=1.):
        super().__init__(dataset, chunk_len, coord_scale, sr_scale)
        self.dataset = dataset
        self.chunk_len = chunk_len
        self.coord_scale = coord_scale
        self.sr_scale = sr_scale
        self.down_srs = None
        self.reinit_down_sr()
        
    def __len__(self):
        return len(self.idx_mapping)

    def reinit_down_sr(self):
        # self.down_srs = [24000, 16000, 12000, 8000]
        self.down_srs = [int(random.random() / 2 * 4800) * 10 + 1000 for _ in range(4)]
        print(self.down_srs)
        # VCTK data is 48k

    def __getitem__(self, idx):
        if idx in self.cached_data.keys():
            return self.cached_data[idx]
        dataset_idx, offset, chunk_len = self.idx_mapping[idx]
        #audio, sr, _, _, _ = self.dataset[idx]
        full_audio, sr = self.dataset[dataset_idx]
        audio = torch.mean(full_audio[:, offset:offset+chunk_len], dim=0).view(1, -1)
        
        down_audio_list = []
        coord_list = []
        for down_sr in self.down_srs:
            down_audio = resize_audio_fn(audio, sr, down_sr)
            down_ratio = down_sr / 48000
            coord = make_coord(down_audio.shape, ranges=[(self.sr_scale * down_ratio, self.sr_scale * down_ratio), (-self.coord_scale, self.coord_scale)])

            down_audio_list.append(down_audio)
            coord_list.append(coord)
        
        down_audio = torch.cat(down_audio_list, dim=1)
        # [channel, t]
        coord = torch.cat(coord_list, dim=0)
        # [t, 2]

        lengths = [c.shape[0] for c in coord_list]
        
        data = {
            'inp': down_audio.view(-1, 1),
            'coord': coord,
            'gt': down_audio.view(-1, 1),
            'index': torch.LongTensor([idx]),
            'offset': torch.LongTensor([0, lengths[0], sum(lengths[0:2]), sum(lengths[0:3])]),
            'len': torch.LongTensor(lengths)
        }
        self.cached_data[idx] = data
        return data


@register('audio-down')
class AudioCoordDownsampled(Dataset):
    def __init__(self, dataset, coord_scale=1., sr_scale=1.):
        super().__init__()
        self.dataset = dataset
        self.coord_scale = coord_scale
        self.sr_scale = sr_scale
        self.down_srs = None
        self.reinit_down_sr()

        self.cached_data = {}
        
    def __len__(self):
        return len(self.dataset)

    def reinit_down_sr(self):
        # self.down_srs = [24000, 16000, 12000, 8000]
        self.down_srs = [int(random.random() / 2 * 4800) * 10 + 1000 for _ in range(4)]
        print(self.down_srs)
        # VCTK data is 48k

    def __getitem__(self, idx):
        if idx in self.cached_data.keys():
            return self.cached_data[idx]
        full_audio, sr = self.dataset[idx]
        audio = torch.mean(full_audio[:, :], dim=0).view(1, -1)
        
        down_audio_list = []
        coord_list = []
        for down_sr in self.down_srs:
            down_audio = resize_audio_fn(audio, sr, down_sr)
            down_ratio = down_sr / 48000
            coord = make_coord(down_audio.shape, ranges=[(self.sr_scale * down_ratio, self.sr_scale * down_ratio), (-self.coord_scale, self.coord_scale)])

            down_audio_list.append(down_audio)
            coord_list.append(coord)
        
        down_audio = torch.cat(down_audio_list, dim=1)
        # [channel, t]
        coord = torch.cat(coord_list, dim=0)
        # [t, 2]

        lengths = [c.shape[0] for c in coord_list]
        
        data = {
            'inp': down_audio.view(-1, 1),
            'coord': coord,
            'gt': down_audio.view(-1, 1),
            'index': torch.LongTensor([idx]),
            'offset': torch.LongTensor([0, lengths[0], sum(lengths[0:2]), sum(lengths[0:3])]),
            'len': torch.LongTensor(lengths)
        }
        self.cached_data[idx] = data
        return data

