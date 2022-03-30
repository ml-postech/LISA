import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio import transforms
import librosa
import numpy as np

import math

from models import register


@register('conv-enc')
class ConvEncoder(nn.Module):
    def __init__(self, latent_dim, in_dim, kernel_size=7, stride=1):
        super().__init__()

        self.latent_dim = latent_dim
        self.in_dim = in_dim

        self.conv_blocks = nn.Sequential(
            nn.Conv1d(in_dim, 16, kernel_size, padding=3, stride=stride),
            nn.Tanh(),
            nn.Conv1d(16, 32, 3, padding=1, stride=1),
            nn.Tanh(),
            nn.Conv1d(32, 64, 3, padding=1, stride=1),
            nn.Tanh(),
            nn.Conv1d(64, latent_dim, 1, padding=0),
        )

    def forward(self, inp):
        assert inp.shape[-2] == self.in_dim
        return self.conv_blocks(inp)
