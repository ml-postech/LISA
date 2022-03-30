import argparse
import os

import torch
import torchaudio


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.wav')
    parser.add_argument('--sr', type=int)
    parser.add_argument('--output', default='output.wav')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    audio, sr = torchaudio.load(args.input)

    resampled_audio = torchaudio.transforms.Resample(sr, args.sr, resampling_method='sinc_interpolation')(audio)

    torchaudio.save(args.output, resampled_audio, args.sr)
