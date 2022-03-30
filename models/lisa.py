import torch
import torch.nn as nn
import torch.nn.functional as F
import random

import models
from models import register
from utils import make_coord


class PositionEmbedding(nn.Module):
    def __init__(self, in_channels, N_freqs=10):
        super(PositionEmbedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)
        
        self.freq_bands = torch.cat([2**torch.linspace(0, N_freqs-1, N_freqs)])

    def forward(self, x):
        out = [x]

        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]
        return torch.cat(out, -1)


# @register('lisa')
class LISA(nn.Module):

    def __init__(self, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True,
                 embedding=None):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        self.imnet = models.make(imnet_spec)
        
        self.embed = None
        if embedding == 'pe':
            in_features = 1
            N_freqs = 6
            total_in_features = N_freqs * 2 * in_features + in_features
            self.embed = PositionEmbedding(in_features, N_freqs)




    def query_rgb(self, coord, latent, cell=None, train=False):
        #coord
        #[B, chunk_len, 1]

        feat = latent
        #[B, local_chunk_num, latent_dim]
        feat = feat.permute(0, 2, 1)
        #[B, latent_dim, local_chunk_num]

        batch_size = feat.shape[0]
        latent_dim = feat.shape[1]
        num_latents = feat.shape[2]
        chunk_len = coord.shape[1]


        # on axis3, stride 1, on both sides
        # rgb query: also 9 points
        if self.feat_unfold:
            feat_prev = torch.cat((feat[:, :, 0:1], feat[:, :, :-1]), dim=2)
            feat_next = torch.cat((feat[:, :, 1:], feat[:, :, -1:]), dim=2)
            feat = torch.cat([feat_prev, feat, feat_next], dim=1)
            # feat = F.unfold(feat, 3, padding=1).view(
            #     feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        var_batch = None
        if train == True:
            var = (random.random() * 2 - 1) * 1
            var_batch = torch.randn((batch_size, chunk_len)).cuda() * 1 - 1
            vx_lst, eps_shift = [-1 + var], 0
        # elif self.local_ensemble:
        else:
            vx_lst = [-2, 0]
            eps_shift = 1e-6
        # else:
        #     vx_lst, eps_shift = [-1], 0

        # field radius (global: [-1, 1])
        rx = 2 / num_latents / 2     # TODO: not coord(with q) and feat

        # x,y coord
        feat_coord = make_coord(feat.shape[-1:], flatten=False).cuda() \
            .view(1, -1) \
            .expand(batch_size, num_latents)
            # .permute(0, 1) \
        # [B, 128]
        # feat_coord = torch.cat([feat_coord, torch.zeros_like(feat_coord)], dim=1)
        # [B, 2, 128]

        preds = []
        areas = []
        # coord = torch.cat([torch.zeros_like(coord), coord], dim=2)
        # [B, q, 2]
        for vx in vx_lst:
            coord_ = coord.clone()
            # coord_ = torch.cat([coord_, torch.zeros_like(coord_)], dim=2)
            if var_batch is not None:
                coord_[:, :, 0] += var_batch * rx
            else:
                coord_[:, :, 0] += vx * rx + eps_shift
            coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
            index = torch.searchsorted(feat_coord[0], coord_.squeeze(2)).clamp(max=num_latents-1)
            q_coord = torch.stack([feat_coord[b, index[b]] for b in range(batch_size)])
            q_coord = q_coord.unsqueeze(2)
            q_feat = torch.stack([feat[b, :, index[b]] for b in range(batch_size)])
            q_feat = q_feat.permute(0, 2, 1)

            rel_coord = coord - q_coord
            rel_coord[:, :, 0] *= num_latents
            # inp = torch.cat([q_feat, rel_coord], dim=-1)

            if self.embed is not None:
                rel_coord = self.embed(rel_coord)


            # if self.cell_decode:
            #     rel_cell = cell.clone()
            #     rel_cell[:, :, 0] *= feat.shape[-2]
            #     inp = torch.cat([inp, rel_cell], dim=-1)

            bs, q = coord.shape[:2]
            pred = self.imnet(rel_coord, q_feat).view(bs, q, -1)
            preds.append(pred)

            area = torch.abs(rel_coord[:, :, 0])
            areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[1]; areas[1] = t

        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def forward(self, coord, latent, cell=None, train=False):
        return self.query_rgb(coord, latent, cell, train)


@register('lisa-gon')
class LISAGON(LISA):

    def __init__(self, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True,
                 enc_loss_fn = 'l2', local_chunk_len=1, latent_dim=16,
                 embedding=None):
        super().__init__(imnet_spec, local_ensemble, feat_unfold, cell_decode, embedding)
        self.gon = models.gon_encoder.GON(loss_fn_name=enc_loss_fn, latent_dim=latent_dim)
        self.local_chunk_len = local_chunk_len
        self.latent_dim = latent_dim

    def forward(self, model_input, train=False):
        gon_coord = model_input['inp_coord'][:, :, 1:2]
        gon_input = model_input['inp']
        coord = model_input['coord'][:, :, 1:2]

        batch_size = coord.shape[0]
        num_latent = int(gon_coord.shape[1] // self.local_chunk_len)

        latent = self.gon(self, gon_coord, gon_input, (batch_size, num_latent, self.latent_dim))
        return self.query_rgb(coord, latent, train=train)
        
    def forward_gon(self, coord, latent):
        assert coord.shape[1] == latent.shape[1]
        
        feat = latent
        #[B, local_chunk_num, latent_dim]
        feat = feat.permute(0, 2, 1)
        #[B, latent_dim, local_chunk_num]

        batch_size = feat.shape[0]
        latent_dim = feat.shape[1]
        num_latents = feat.shape[2]
        chunk_len = coord.shape[1]
        in_dim = coord.shape[2]

        local_context = 3 

        # on axis3, stride 1, on both sides
        # rgb query: also 9 points
        if self.feat_unfold:
            feat_prev = torch.cat((feat[:, :, 0:1], feat[:, :, :-1]), dim=2)
            feat_next = torch.cat((feat[:, :, 1:], feat[:, :, -1:]), dim=2)
            feat = torch.cat([feat_prev, feat, feat_next], dim=1)
            # feat = F.unfold(feat, 3, padding=1).view(
            #     feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        feat_local = feat.unsqueeze(3).repeat(1, 1, 1, 2 * local_context + 1)
        #[B, latent_dim, num_latents, 2*local_context + 1]

        #[B, chunk_len, C]
        coord = coord.permute(0, 2, 1)
        coord_locals_prev = []
        coord_locals_next = []
        for i in range(1, local_context + 1):
            coord_tmp = torch.cat([coord[:, :, :1,]] * i + [coord[:, :, :-i]], dim=2)
            coord_locals_prev.append(coord_tmp)
            coord_tmp = torch.cat([coord[:, :, i:]] + [coord[:, :, -1:]] * i, dim=2)
            coord_locals_next.append(coord_tmp)
        coord_locals_prev.reverse()
        coord_local = torch.stack(coord_locals_prev + [coord] + coord_locals_next, dim=3)
        #[B, C, chunk_len, 2*local_context + 1]

        coord_center = coord.unsqueeze(3).repeat(1, 1, 1, 2 * local_context + 1)
        rel_coord = coord_local - coord_center
        rel_coord[:, 0, :, :] *= num_latents

        if self.feat_unfold:
            feat_local = feat_local.view(batch_size, 3 * latent_dim, -1).permute(0, 2, 1)
        else:
            feat_local = feat_local.view(batch_size, latent_dim, -1).permute(0, 2, 1)
        rel_coord = rel_coord.view(batch_size, in_dim, -1).permute(0, 2, 1)
        if self.embed is not None:
            rel_coord = self.embed(rel_coord)
        pred = self.imnet(rel_coord, feat_local).view(batch_size, chunk_len, 2 * local_context + 1, 1)

        # TODO: chunk 끝 부분이 원래보다 0에 가까움
        pred_splits = [] 
        for i in range(1, local_context + 1):
            pred_splits.append(torch.cat([pred[:, i:, i-1, :], torch.zeros((batch_size, i, 1)).cuda()], dim=1))
        pred_splits.reverse()
        pred_splits.append(pred[:, :, local_context, :])
        for i in range(1, local_context + 1):
            pred_splits.append(torch.cat([torch.zeros((batch_size, i, 1)).cuda(), pred[:, :-i, local_context + i, :]], dim=1))
        pred = torch.mean(torch.stack(pred_splits, dim=3), dim=3)

        return pred

@register('lisa-enc')
class LISAEncoder(LISA):

    def __init__(self, imnet_spec, encoder_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True,
                 latent_dim=32, embedding=None,
                 ):
        super().__init__(imnet_spec, local_ensemble, feat_unfold, cell_decode, embedding)
        if encoder_spec is None:
            self.encoder = models.audio_encoder.ConvEncoder(latent_dim, 1)
        else:
            self.encoder = models.make(encoder_spec)
        self.latent_dim = latent_dim

    def forward(self, model_input, train=False):
        inp = model_input['inp']
        coord = model_input['coord'][:, :, 1:2]

        batch_size = coord.shape[0]

        latent = self.encoder(inp.view(batch_size, 1, -1)).permute(0,2,1)
        return self.query_rgb(coord, latent, train=train)
