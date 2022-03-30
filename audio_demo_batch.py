import argparse
import os

import torch
import torchaudio

import models
from utils import make_coord
# from test import batched_predict

from datasets import audio_dataset, wrappers


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.wav')
    parser.add_argument('--model')
    parser.add_argument('--sr', type=int)
    parser.add_argument('--output', default='output.wav')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    root_path = args.input

    saved = torch.load(args.model)
    model = models.make(saved['model'], load_sd=True).cuda()
    model.eval()
    # saved_latent = saved['latent_list']
    # latent = torch.stack(saved_latent).cuda()           # TODO

    sinlge_dataset = audio_dataset.AudioSingle(root_path)
    dataset = wrappers.AudioCoordChunked(dataset=sinlge_dataset, 
                                    chunk_len=0.02, 
                                    coord_scale=1,
                                    gt_sr=16000)

    audio_len = dataset.idx_mapping[-1][1] + dataset.idx_mapping[-1][2]
    result_audio = torch.zeros((1, audio_len)).cuda()

    batch_size = 1
    latent_dim = 16
    enc_loss_fn = torch.nn.MSELoss()

    chunk_num = len(dataset)
    for i in range(chunk_num):
        batch = dataset[i]
        for k, v in batch.items():
            batch[k] = v.cuda()

        ####

        z0 = torch.zeros((batch_size, latent_dim)).cuda().requires_grad_(True)
        coord_ = torch.cat([z0.unsqueeze(1).repeat(1, batch['coord'].shape[0], 1), batch['coord'][:, 1:2].unsqueeze(0)], dim=2)
        pred_ = model(coord_)
        loss1 = enc_loss_fn(pred_, batch['gt'].unsqueeze(0))
        grad = torch.autograd.grad(loss1, [z0], create_graph=True, retain_graph=True)[0]
        # z1 = -grad * batch_size
        latents = -grad * batch_size
    
        coord_ = torch.cat([latents.unsqueeze(1).repeat(1, batch['coord'].shape[0], 1), batch['coord'][:, 1:2].unsqueeze(0)], dim=2)
        pred = model(coord_)[0]
        pred = pred.clamp(-1, 1).view(1, -1)
        ####

        curr_offset = dataset.idx_mapping[i][1]
        curr_len = dataset.idx_mapping[i][2]
        if i > 0:
            prev_offset = dataset.idx_mapping[i-1][1]
            prev_len = dataset.idx_mapping[i-1][2]
            overlap_len = prev_offset + prev_len - curr_offset
            if overlap_len > 0:
                overlap_mask = torch.arange(overlap_len + 2).cuda() / (overlap_len + 2)
                overlap_mask = overlap_mask[1:-1]       # because of when overlap_len == 1

                pred[0, :overlap_len] *= overlap_mask
                assert overlap_len * 2 < curr_len  # TODO: no overlapping 3 chunks

        if i < chunk_num - 1:
            next_offset = dataset.idx_mapping[i+1][1]
            next_len = dataset.idx_mapping[i+1][2]
            overlap_len = curr_offset + curr_len - next_offset
            if overlap_len > 0:
                overlap_mask = torch.arange(overlap_len + 2).cuda() / (overlap_len + 2)
                overlap_mask = torch.flip(overlap_mask[1:-1], [0])

                pred[0, -overlap_len:] *= overlap_mask
                assert overlap_len * 2 < curr_len
        
        result_audio[:, curr_offset:curr_offset+curr_len] += pred

    pred_long = result_audio.view(1, -1).cpu()
    
    torchaudio.save(args.output, pred_long, args.sr)
