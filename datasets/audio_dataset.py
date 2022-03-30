import os
import json

import pickle
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

# from torchaudio import datasets.VCTK_092 as VCTK_092
# import torchaudio.datasets.VCTK_092 as VCTK_092

from datasets import register


@register('audio-folder')
class AudioFolder(Dataset):

    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none'):
        self.repeat = repeat

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)

            self.files.append(file)


    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]
        waveform, sr = torchaudio.load(x)
        #audio_chunk = torch.mean(audio[:, :sr * 2], dim=0)

        return waveform, sr  #audio_chunk,


@register('VCTK')
class VCTK(torchaudio.datasets.VCTK):

    exception_list = ['p306_151', 'p323_424', 'p361_173', 'p363_354']

    def __init__(self, root_path, download=False, split=''):
        super(VCTK, self).__init__(root_path, download=download)
        self._walker = list(filter(lambda w: w not in self.exception_list, self._walker))
        if split == 'train':
            self._walker = list(filter(lambda w: int(w[1:4]) < 350, self._walker))
        if split == 'val':
            self._walker = list(filter(lambda w: int(w[1:4]) >= 350, self._walker))

    
    def __getitem__(self, idx):
        waveform, sr, _, _, _ = super().__getitem__(idx)
        return waveform, sr
    
    
@register('VCTK-small')
class VCTKSmall(torchaudio.datasets.VCTK):

    exception_list = ['p306_151', 'p323_424', 'p361_173', 'p363_354']
    val_list = ['p374_001', 'p374_002', 'p376_001', 'p376_002']

    def __init__(self, root_path, download=False, split=''):
        super(VCTKSmall, self).__init__(root_path, download=download)
        self._walker = list(filter(lambda w: w not in self.exception_list, self._walker))
        if split == 'train':
            self._walker = list(filter(lambda w: int(w[1:4]) < 229, self._walker))
        if split == 'val':
            self._walker = list(filter(lambda w: w in self.val_list, self._walker))

    
    def __getitem__(self, idx):
        waveform, sr, _, _, _ = super().__getitem__(idx)
        return waveform, sr
    
@register('VCTK-xs')
class VCTKXsmall(torchaudio.datasets.VCTK):

    exception_list = ['p306_151', 'p323_424', 'p361_173', 'p363_354']
    # val_list = ['p225_359', 'p225_363']
    val_list = ['p225_365', 'p225_366']

    def __init__(self, root_path, download=False, split=''):
        super(VCTKXsmall, self).__init__(root_path, download=download)
        self._walker = list(filter(lambda w: w not in self.exception_list, self._walker))
        if split == 'train':
            self._walker = list(filter(lambda w: w[1:4] == '225' and int(w[5:8]) < 365, self._walker))
        if split == 'val':
            self._walker = list(filter(lambda w: w in self.val_list, self._walker))

    
    def __getitem__(self, idx):
        waveform, sr, _, _, _ = super().__getitem__(idx)
        return waveform, sr
    
    
@register('VCTK-xxs')
class VCTKXXsmall(torchaudio.datasets.VCTK):

    exception_list = ['p306_151', 'p323_424', 'p361_173', 'p363_354']

    def __init__(self, root_path, download=False, split=''):
        super(VCTKXXsmall, self).__init__(root_path, download=download)
        self._walker = list(filter(lambda w: w not in self.exception_list, self._walker))
        # if split == 'train':
        self._walker = list(filter(lambda w: w[1:4] == '225', self._walker))[:10]
        if split == 'val':
            self._walker = self._walker[-2:]

    
    def __getitem__(self, idx):
        waveform, sr, _, _, _ = super().__getitem__(idx)
        return waveform, sr
    

@register('VCTK-single')
class VCTKSingle(torchaudio.datasets.VCTK):

    def __init__(self, root_path, download=False):
        super(VCTKSingle, self).__init__(root_path, download=download)
        self._walker = list(filter(lambda w: w == 'p306_205', self._walker))

    def __getitem__(self, idx):
        waveform, sr, _, _, _ = super().__getitem__(idx)
        return waveform, sr
    

@register('audio-single')
class AudioSingle():

    def __init__(self, root_path):
        self.filename = root_path
        self.waveform, self.sr = torchaudio.load(self.filename)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.filename)
        return waveform, sr
    
    def __len__(self):
        return 1



        


