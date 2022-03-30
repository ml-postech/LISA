import math
import torch
from torch import nn
import torch.nn.functional as F
import models
from models import register
import utils



@register('gon')
class GON(nn.Module):
    def __init__(self, loss_fn_name='l2', latent_dim=None):
        super().__init__()
        self.loss_fn = utils.get_loss_fn(loss_fn_name)
        self.latent_dim = latent_dim

    def forward(self, model, coord, input, latent_shape=None):
        # model should have forward_gon(coord, latent)
        batch_size = coord.shape[0]

        if latent_shape is None:
            z0 = torch.zeros((batch_size, self.latent_dim)).cuda().requires_grad_(True)
        else:
            z0 = torch.zeros(latent_shape).cuda().requires_grad_(True)
        pred = model.forward_gon(coord, z0)
        loss1, _ = self.loss_fn(pred, input)
        grad = torch.autograd.grad(loss1, [z0], create_graph=True, retain_graph=True)[0]
        latents = -grad 
        latents = latents / latents.norm(2, dim=-1, keepdim=True) * math.sqrt(self.latent_dim)
        return latents


