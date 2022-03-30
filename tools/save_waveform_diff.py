import argparse
import os

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

import librosa
import matplotlib.pyplot as plt


def plot_waveform_diff(original,pred, sample_rate, filename, title="Waveform", xlim=None, ylim=None):
  original = original.numpy()
  pred = pred.numpy()
  waveform = (original-pred)

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate
  last = time_axis[-1]
  max_v = max(original.reshape(-1))
  min_v = min(original.reshape(-1))

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
  plt.axis([-0.025,last+0.025, min_v, max_v])
  plt.show(block=False)
  plt.savefig(filename)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', default='original.wav')
    parser.add_argument('--predict', default='pred.wav')
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--time1', type=float, default=0.5)
    parser.add_argument('--time2', type=float, default=1)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    waveform1, sr1 = torchaudio.load(args.original)
    waveform2, sr2 = torchaudio.load(args.predict)
    
    # waveform2, sr2 = torchaudio.load(args.input2)

    # diff = waveform2 - waveform1
    # print(diff.shape)
    # diff = diff[0.5 * sr1:1 * sr1]

    print(waveform1.shape)
    waveform1 = waveform1[:, int(args.time1 * sr1):int(args.time2 * sr1)]
    waveform2 = waveform2[:, int(args.time1 * sr1):int(args.time2 * sr1)]

    plot_waveform_diff(waveform1,waveform2, sr1, filename=args.output, title="")
