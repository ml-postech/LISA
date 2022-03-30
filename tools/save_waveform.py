import argparse
import os

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

import librosa
import matplotlib.pyplot as plt


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




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.wav')
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--time1', type=float, default=0.5)
    parser.add_argument('--time2', type=float, default=1)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    waveform1, sr1 = torchaudio.load(args.input)
    # waveform2, sr2 = torchaudio.load(args.input2)

    # diff = waveform2 - waveform1
    # print(diff.shape)
    # diff = diff[0.5 * sr1:1 * sr1]

    print(waveform1.shape)
    waveform1 = waveform1[:, :]

    plot_waveform(waveform1, sr1, filename=args.output, title="", xlim=[args.time1, args.time2])
