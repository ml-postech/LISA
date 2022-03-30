import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

import models
from models import register



# helpers
def exists(val):
    return val is not None

def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)


# sin activation
class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)


# siren layer
class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0 = 0.2, c = 6., is_first = False, use_bias = True, activation = None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0) 
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out =  F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out


# siren network
@register('siren')
class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0 = 0.2, w0_initial = 30., use_bias = True, final_activation = None, relu=False):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden
            activation = nn.ReLU() if relu else None

            self.layers.append(Siren(
                dim_in = layer_dim_in,
                dim_out = dim_hidden,
                w0 = layer_w0,
                use_bias = use_bias,
                is_first = is_first,
                activation = activation
            ))

        final_activation = nn.Identity() if not exists(final_activation) else final_activation
        self.last_layer = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation)

    def forward(self, coord, mods = None):
        x = coord[:, :, :]
        mods = cast_tuple(mods, self.num_layers)

        for layer, mod in zip(self.layers, mods):
            x = layer(x)

            if exists(mod):
                if len(mod.shape) == 2:
                    x *= mod.unsqueeze(1)
                else:
                    x *= mod
                # x *= rearrange(mod, 'd -> () d')

        return self.last_layer(x)


# modulatory feed forward
class Modulator(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([])

        
        for ind in range(num_layers):
            is_first = ind == 0
            dim = dim_in if is_first else (dim_hidden + dim_in) ## 왜 dim_in을 더해주는 거지?

            self.layers.append(nn.Sequential(
                nn.Linear(dim, dim_hidden),
                nn.ReLU()
            ))

    def forward(self, z):
        assert len(z.shape) == 2 or len(z.shape) == 3
        x = z
        hiddens = []

        for layer in self.layers:
            x = layer(x)
            hiddens.append(x)
            x = torch.cat((x, z), dim=-1)

        return tuple(hiddens)


# siren network
@register('siren+')
class SirenNetPlus(SirenNet):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, dim_latent=16, w0 = 0.2, w0_initial = 30., use_bias = True, final_activation = None, relu = False):
        super().__init__(dim_in + dim_latent, dim_hidden, dim_out, num_layers, w0, w0_initial, use_bias, final_activation, relu)
        
    def forward(self, coord, latent, mods = None):
        if len(latent.shape) == 3:
            x = torch.cat([coord[:, :, :], latent], dim=2)
        else:
            l = latent.unsqueeze(1).repeat(1, coord.shape[1], 1)
            x = torch.cat([coord[:, :, :], l], dim=2)
        mods = cast_tuple(mods, self.num_layers)

        for layer, mod in zip(self.layers, mods):
            x = layer(x)

            if exists(mod):
                if len(mod.shape) == 2:
                    x *= mod.unsqueeze(1)
                else:
                    x *= mod
                # x *= rearrange(mod, 'd -> () d')

        return self.last_layer(x)


# wrapper
@register('mod-sine')
class ModSine(nn.Module):
    def __init__(self, 
            latent_dim=1024, 
            dim_in=1,
            dim_out=1,
            dim_hidden=1024, # TODO: remove
            num_layers=7,
            w0_initial=30.,
            w0=0.2,
            siren_plus=False,
            ):
        super().__init__()
        
        self.siren_plus = siren_plus

        if self.siren_plus:
            net = SirenNetPlus(     
                dim_in = dim_in,            # input dimension, ex. 2d coor
                dim_hidden = dim_hidden,                 # hidden dimension
                dim_out = dim_out,                       # output dimension, ex. rgb value
                num_layers = num_layers,                 # number of layers
                w0_initial = w0_initial,                 # different signals may require different omega_0 in the first layer - this is a hyperparameter
                dim_latent = latent_dim,
                w0 = w0,                                 # different signals may require different omega_0 in the first layer - this is a hyperparameter
            )
        else:
            net = SirenNet(     
                dim_in = dim_in,                         # input dimension, ex. 2d coor
                dim_hidden = dim_hidden,                 # hidden dimension
                dim_out = dim_out,                       # output dimension, ex. rgb value
                num_layers = num_layers,                 # number of layers
                w0_initial = w0_initial,                 # different signals may require different omega_0 in the first layer - this is a hyperparameter
                w0 = w0,                                 # different signals may require different omega_0 in the first layer - this is a hyperparameter
            )

        self.net = net
        self.latent_dim = latent_dim

        self.modulator = Modulator(
            dim_in = latent_dim,
            dim_hidden = dim_hidden,
            num_layers = num_layers
        )

    def forward(self, coord, latent):
        mods = self.modulator(latent)
        if self.siren_plus:
            out = self.net(coord, latent, mods=mods)        
        else:
            out = self.net(coord, mods=mods)        
        return out


@register('mod-sine-enc')
class ModSineEncoder(nn.Module):
    def __init__(self, 
            encoder_spec,
            latent_dim = None, 
            dim_in=1,
            dim_out=1,
            dim_hidden=1024,
            num_layers=7,
            w0_initial=30.,
            w0=0.2,
            ):
        super().__init__()

        net = SirenNet(     # TODO: read from config
            dim_in = dim_in,                         # input dimension, ex. 2d coor
            dim_hidden = dim_hidden,                 # hidden dimension
            dim_out = dim_out,                       # output dimension, ex. rgb value
            num_layers = num_layers,                 # number of layers
            w0_initial = w0_initial,                 # different signals may require different omega_0 in the first layer - this is a hyperparameter
            w0 = w0,                                 # different signals may require different omega_0 in the first layer - this is a hyperparameter
        )

        self.net = net
        self.latent_dim = latent_dim

        self.encoder = models.make(encoder_spec)

        self.modulator = Modulator(
            dim_in = latent_dim,
            dim_hidden = net.dim_hidden,
            num_layers = net.num_layers
        )

    def forward(self, inp, coord):
        batch_size = inp.shape[0]
        latent = self.encoder(inp)
        # [latent_dim, 1]
        mods = self.modulator(latent.view(batch_size, -1))
        out = self.net(coord, mods=mods)        

        return out
        