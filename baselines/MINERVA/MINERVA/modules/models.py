from os import path
from os.path import join as pj
import math
import copy
import numpy as np
import torch as th
import torch.nn as nn

import functions.models as F
import modules.utils as utils


class Net(nn.Module):
    def __init__(self, o):
        super(Net, self).__init__()
        self.o = o
        self.sct = SCT(o)
        self.loss_calculator = LossCalculator(o)

    def forward(self, inputs):
        x_r_pre, s_r_pre, z_mu, z_logvar, z, c, b, z_uni, c_all, z_mix = self.sct(inputs)
        sum_losses, total_loss, raw_loss = self.loss_calculator(inputs, x_r_pre, s_r_pre, z_mu, z_logvar, z, c, b, z_uni, z_mix)
        
        return sum_losses, total_loss, raw_loss, c_all


class SCT(nn.Module):
    def __init__(self, o):
        super(SCT, self).__init__()
        self.o = o
        self.sampling = False
        self.batch_correction = False
        self.b_centroid = None

        # Modality encoders q(z|x^m)
        x_encs = {}
        if len(o.dims_enc_x) > 0:
            x_shared_enc = MLP(o.dims_enc_x+[o.dim_z*2], hid_norm=o.norm, hid_drop=o.drop)
            for m in o.ref_mods:
                x_indiv_enc = MLP([o.dims_h[m], o.dims_enc_x[0]], out_trans='mish', norm=o.norm,
                                drop=o.drop)
                x_encs[m] = nn.Sequential(x_indiv_enc, x_shared_enc)
        else:
            for m in o.ref_mods:
                x_encs[m] = MLP([o.dims_h[m], o.dim_z*2], hid_norm=o.norm, hid_drop=o.drop)
        self.x_encs = nn.ModuleDict(x_encs)
        
        # Modality decoder p(x^m|c, b)
        self.x_dec = MLP([o.dim_z]+o.dims_dec_x+[sum(o.dims_h.values())], hid_norm=o.norm,
                         hid_drop=o.drop)

        # Batch encoder q(z|s)
        self.s_enc = MLP([o.dim_s]+o.dims_enc_s+[o.dim_z*2], hid_norm=o.norm, hid_drop=o.drop)
        # Batch decoder p(s|b)
        self.s_dec = MLP([o.dim_b]+o.dims_dec_s+[o.dim_s], hid_norm=o.norm, hid_drop=o.drop)


    def forward(self, inputs):

        o = self.o
        s = None
        
        if "s" in inputs.keys():
            s_drop_rate = o.s_drop_rate if self.training else 0
            if th.rand([]).item() < 1 - s_drop_rate:
                s = inputs["s"]
           
        # z_x_mu{"raw":{"rna": tensor([[ ]]), "adt": tensor([[ ]])}, "fusion": {"rna": tensor([[ ]]), "adt": tensor([[ ]])}}
        z_x_mu, z_x_logvar = {}, {}
        z_s_mu, z_s_logvar = {}, {}
        z_mu, z_logvar = {}, {}
        z, c, b = {}, {}, {}
        z_uni, c_all = {}, {}
        x_r_pre, s_r_pre = {}, {}
        z_uni_mu, z_uni_logvar = {}, {}
                
        for t in o.pretext:

            if t in inputs:
                if t in ["raw", "mask", "noise", "downsample"]:
                    z_x_mu[t], z_x_logvar[t] = {}, {}
                    z_uni[t], c_all[t] = {}, {}
                    x = inputs[t]
                    e = inputs["e"]
                    if t in ["noise", "downsample"]:
                        s = None
                    
                else:
                    z_x_mu[t], z_x_logvar[t] = {}, {}
                    z_uni[t], c_all[t] = {}, {}
                    x = inputs[t]
                    if t == "fusion":
                        s = None
                        z_xmix_mu, z_xmix_logvar = {}, {}
                        z_smix_mu, z_smix_logvar = {}, {}
                        z_mix_mu, z_mix_logvar = {}, {}
                        x_mix, s_mix, z_mix = {}, {}, {}

                        for p in ["mix1", "mix2"]:
                            x_mix[p] = inputs[p]
                            z_xmix_mu[p], z_xmix_logvar[p] = {}, {}
                            z_smix_mu[p], z_smix_logvar[p] = {}, {}

                    else:
                        break
            
            else: continue

            # Encode x_m
            x_pp = {}
            for m in x.keys():
                x_pp[m] = preprocess(x[m], m, o.dims_x[m], o.task)

                if t in ["raw", "mask", "noise"]:
                    if m in ["rna", "adt"]:  # use mask
                        h = x_pp[m] * e[m]
                else:
                    h = x_pp[m]
                
                # encoding
                z_x_mu[t][m], z_x_logvar[t][m] = self.x_encs[m](h).split(o.dim_z, dim=1)
                    
                if t == "fusion":
                    for p in ["mix1", "mix2"]:
                        z_xmix_mu[p][m], z_xmix_logvar[p][m]  = self.x_encs[m](th.log1p(x_mix[p][m])).split(o.dim_z, dim = 1)

            # Encode s
            if s is not None: 
                s_pp = nn.functional.one_hot(s["joint"].squeeze(1), num_classes=o.dim_s).float()  # N * B
                
                z_s_mu[t], z_s_logvar[t] = self.s_enc(s_pp).split(o.dim_z, dim=1)
                z_s_mu[t], z_s_logvar[t] = [z_s_mu[t]], [z_s_logvar[t]]

            else:
                z_s_mu[t], z_s_logvar[t] = [], []

            # Use product-of-experts
            z_x_mu_list = list(z_x_mu[t].values())
            z_x_logvar_list = list(z_x_logvar[t].values())
            z_mu[t], z_logvar[t] = poe(z_x_mu_list+z_s_mu[t], z_x_logvar_list+z_s_logvar[t])  # N * K
            
            if t == "fusion":
                for p in ["mix1", "mix2"]:
                    z_smix_mu[p], z_smix_logvar[p] = [], []
                    z_mix_mu[p], z_mix_logvar[p] = poe(list(z_xmix_mu[p].values()) + z_smix_mu[p], list(z_xmix_logvar[p].values()) + z_smix_logvar[p])

            # Sample z
            if self.training:
                z[t] = utils.sample_gaussian(z_mu[t], z_logvar[t])
                if t == "fusion":
                    for p in ["mix1", "mix2"]:
                        z_mix[p] = utils.sample_gaussian(z_mix_mu[p], z_mix_logvar[p])
                else:
                    z_mix = None

            elif self.sampling and o.sample_num > 0:
                z_mu_expand, z_logvar_expand = {}, {}
                z_mu_expand[t] = z_mu[t].unsqueeze(1)  # N * 1 * K
                z_logvar_expand[t] = z_logvar[t].unsqueeze(1).expand(-1, o.sample_num, o.dim_z)  # N * I * K
                z[t] = utils.sample_gaussian(z_mu_expand[t], z_logvar_expand[t]).reshape(-1, o.dim_z)  # NI * K
            else:  # validation
                z[t] = z_mu[t]
                z_mix = None

            c[t], b[t] = z[t].split([o.dim_c, o.dim_b], dim=1)
            
            # Generate x_m activation/probability
            if self.batch_correction:
                if t == "raw":
                    z_bc = z[t].clone()
                    z_bc[:, o.dim_c:] = self.b_centroid.type_as(z[t]).unsqueeze(0)
                    x_r_pre[t] = self.x_dec(z_bc).split(list(o.dims_h.values()), dim=1)
            else:
                x_r_pre[t] = self.x_dec(z[t]).split(list(o.dims_h.values()), dim=1)
            x_r_pre[t] = utils.get_dict(o.ref_mods, x_r_pre[t])
            
            # Generate s activation
            if s is not None:
                s_r_pre[t] = self.s_dec(b[t])
            else:
                s_r_pre[t] = None
                
            for m in z_x_mu[t].keys():
                # Calculate q(z|x^m, s)
                z_uni_mu[t], z_uni_logvar[t] = poe([z_x_mu[t][m]]+z_s_mu[t], [z_x_logvar[t][m]]+z_s_logvar[t])  # N * K
                z_uni[t][m] = utils.sample_gaussian(z_uni_mu[t], z_uni_logvar[t])  # N * K
                c_all[t][m] = z_uni[t][m][:, :o.dim_c]  # N * C
            c_all[t]["joint"] = c[t]


        return x_r_pre, s_r_pre, z_mu, z_logvar, z, c, b, z_uni, c_all, z_mix


def poe(mus, logvars):
    """
    Product of Experts
    - mus: [mu_1, ..., mu_M], where mu_m is N * K
    - logvars: [logvar_1, ..., logvar_M], where logvar_m is N * K
    """
    
    mus = [th.full_like(mus[0], 0)] + mus
    logvars = [th.full_like(logvars[0], 0)] + logvars
    
    mus_stack = th.stack(mus, dim=1)  # N * M * K
    logvars_stack = th.stack(logvars, dim=1)
    
    T = exp(-logvars_stack)  # precision of i-th Gaussian expert at point x
    T_sum = T.sum(1)  # N * K
    pd_mu = (mus_stack * T).sum(1) / T_sum
    pd_var = 1 / T_sum
    pd_logvar = log(pd_var)
    return pd_mu, pd_logvar  # N * K


def gen_real_data(x_r_pre, sampling=True):
    """
    Generate real data using x_r_pre
    - sampling: whether to generate discrete samples
    """
    x_r = {}
    for m, v in x_r_pre.items():
        if m in ["rna", "adt"]:
            x_r[m] = v.exp()
            if sampling:
                x_r[m] = th.poisson(x_r[m]).int()

    return x_r


class Discriminator(nn.Module):

    def __init__(self, o):
        super(Discriminator, self).__init__()
        self.o = o

        predictors = {}
        mods = o.mods + ["joint"]
        for m in mods:
            predictors[m] = MLP([o.dim_c]+o.dims_discriminator+[o.dims_s[m]],
                                    hid_norm=o.norm, hid_drop=o.drop)
        self.predictors = nn.ModuleDict(predictors)
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='sum')  # log_softmax + nll
        
        self.epoch = None
        
        
    def forward(self, c_all, s_all):
        o = self.o
        loss_dict = {}

        for m in s_all.keys():
            c, s = c_all["raw"][m], s_all[m]
            s_r_pre = self.predictors[m](c)
            loss_dict[m] = self.cross_entropy_loss(s_r_pre, s.squeeze(1))

        loss = sum(loss_dict.values()) / c_all["raw"]["joint"].size(0)

        loss = loss * 100

        return loss



class LossCalculator(nn.Module):

    def __init__(self, o):
        super(LossCalculator, self).__init__()
        self.o = o
        self.pois_loss = nn.PoissonNLLLoss(full=True, reduction='none')
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')  # log_softmax + nll
        self.mse_loss = nn.MSELoss(reduction='none')
        self.kld_loss = nn.KLDivLoss(reduction='sum')
        self.gaussian_loss = nn.GaussianNLLLoss(full=True, reduction='sum')

        self.tech_ib_coef = 4


    def forward(self, inputs, x_r_pre, s_r_pre, z_mu, z_logvar, z, c, b, z_uni, z_mix = None):
        o = self.o
                
        losses, loss_ratio = {}, {}
        loss_recon, loss_kld_z, loss_mod, loss_SEM = {}, {}, {}, {}

        for t in z_uni.keys():

            losses[t], loss_ratio[t] = {}, {}
            loss_recon[t], loss_kld_z[t], loss_mod[t], loss_SEM["fusion"] = {}, {}, {}, {}

            
            if t in ["raw", "mask", "noise", "downsample"]:
                e = inputs["e"]
                x = inputs["raw"]
                s = inputs["s"]["joint"]

            if t == "fusion":
                x = inputs[t]
                s = inputs["s"]["joint"]

            loss_recon[t] = self.calc_recon_loss(t, x, s, e, x_r_pre[t], s_r_pre[t])
            loss_kld_z[t] = self.calc_kld_z_loss(z_mu[t], z_logvar[t])

            loss_mod[t] = self.calc_mod_align_loss(z_uni[t])
            loss_mod[t] = loss_mod[t] * 50

            if o.debug == 1:
                print("%s=> recon: %.3f\tkld_z: %.3f\ttopo: %.3f" % (t, loss_recon[t].item(),
                    loss_kld_z[t].item(), loss_mod[t].item()))
            
        if "fusion" in z_mu.keys():
            lam = inputs["mix_param"][:,2].view(-1, 1).cuda()
            loss_SEM["fusion"] = self.consistency_loss(lam, z, z_mix)

            sum_losses = {
                "loss_recon": loss_recon,
                "loss_kld_z": loss_kld_z,
                "loss_mod": loss_mod,
                "loss_SEM": loss_SEM,
                }
            
        else:
             sum_losses = {
                "loss_recon": loss_recon,
                "loss_kld_z": loss_kld_z,
                "loss_mod": loss_mod,
                }

        total_loss = 0
        raw_loss = 0

        for index, (outer_key, inner_dict) in enumerate(sum_losses.items()):
            
            for value1 in inner_dict.values():
                total_loss += value1
            raw_loss += list(inner_dict.values())[0]
    
        return sum_losses, total_loss, raw_loss


    def calc_recon_loss(self, t, x, s, e, x_r_pre, s_r_pre):
        o = self.o
        losses = {}
        # Reconstruciton losses of x^m
        for m in x.keys():
            if m == "label":
                losses[m] = self.cross_entropy_loss(x_r_pre[m], x[m].squeeze(1)).sum()
            else:
                if t == "raw":
                    losses[m] = (self.pois_loss(x_r_pre[m], x[m]) * e[m]).sum()
                elif t in ["fusion"]:
                    losses[m] = (self.pois_loss(x_r_pre[m], x[m])).sum() * 0.01
                    # print(losses[m])
                elif t in ["noise", "downsample"]:
                    losses[m] = (self.pois_loss(x_r_pre[m], x[m])).sum() * 0.05
                else:
                    losses[m] = (self.pois_loss(x_r_pre[m], x[m]) * e[m]).sum() * 0.25
                    
        if s_r_pre is not None:
            s_coef = 1000
            s_coef_mix = 1000

            if t == "fusion":
                losses["s"] = self.cross_entropy_loss(s_r_pre, s.squeeze(1)).sum() * (self.tech_ib_coef + s_coef_mix)
            else:
                losses["s"] = self.cross_entropy_loss(s_r_pre, s.squeeze(1)).sum() * (self.tech_ib_coef + s_coef)

        # print(losses)
        return sum(losses.values()) / s.size(0)


    def calc_kld_z_loss(self, mu, logvar):
        o = self.o
        mu_c, mu_b = mu.split([o.dim_c, o.dim_b], dim=1)
        logvar_c, logvar_b = logvar.split([o.dim_c, o.dim_b], dim=1)
        kld_c_loss = self.calc_kld_loss(mu_c, logvar_c)
        kld_b_loss = self.calc_kld_loss(mu_b, logvar_b)
        kld_z_loss = kld_c_loss + (1 + self.tech_ib_coef) * kld_b_loss
        return kld_z_loss


    def calc_kld_loss(self, mu, logvar):
        return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum() / mu.size(0)


    def calc_mod_align_loss(self, z_uni):
        z_uni_stack = th.stack(list(z_uni.values()), dim=0)  # M * N * K
        z_uni_mean = z_uni_stack.mean(0, keepdim=True)  # 1 * N * K
        return ((z_uni_stack - z_uni_mean)**2).sum() / z_uni_stack.size(1)
    

    def consistency_loss(self, lam, z, z_mix):
        z_Mix = lam * z_mix["mix1"] + (1 - lam) * z_mix["mix2"]
        con_loss = self.mse_loss(z_Mix, z["fusion"]).sum()
        return con_loss * 0.1


class MLP(nn.Module):
    def __init__(self, features=[], hid_trans='mish', out_trans=False,
                 norm=False, hid_norm=False, drop=False, hid_drop=False):
        super(MLP, self).__init__()
        layer_num = len(features)
        assert layer_num > 1, "MLP should have at least 2 layers!"
        if norm:
            hid_norm = out_norm = norm
        else:
            out_norm = False
        if drop:
            hid_drop = out_drop = drop
        else:
            out_drop = False
        
        layers = []
        for i in range(1, layer_num):
            layers.append(nn.Linear(features[i-1], features[i]))
            if i < layer_num - 1:  # hidden layers (if layer number > 2)
                layers.append(Layer1D(features[i], hid_norm, hid_trans, hid_drop))
            else:                  # output layer
                layers.append(Layer1D(features[i], out_norm, out_trans, out_drop))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    

class Layer1D(nn.Module):
    def __init__(self, dim=False, norm=False, trans=False, drop=False):
        super(Layer1D, self).__init__()
        layers = []
        if norm == "bn":
            layers.append(nn.BatchNorm1d(dim))
        elif norm == "ln":
            layers.append(nn.LayerNorm(dim))
        if trans:
            layers.append(func(trans))
        if drop:
            layers.append(nn.Dropout(drop))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def preprocess(x, name, dim, task):
    if name == "label":
        x = nn.functional.one_hot(x.squeeze(1), num_classes=dim).float()
    elif name == "rna":
        x = x.log1p()
    elif name == "adt":
        x = x.log1p()
    return x


def norm_grad(input, max_norm):
    if input.requires_grad:
        def norm_hook(grad):
            N = grad.size(0)  # batch number
            norm = grad.view(N, -1).norm(p=2, dim=1) + 1e-6
            scale = (norm / max_norm).clamp(min=1).view([N]+[1]*(grad.dim()-1))
            return grad / scale

            # clip_coef = float(max_norm) / (grad.norm(2).data[0] + 1e-6)
            # return grad.mul(clip_coef) if clip_coef < 1 else grad
        input.register_hook(norm_hook)


def clip_grad(input, value):
    if input.requires_grad:
        input.register_hook(lambda g: g.clamp(-value, value))


def scale_grad(input, scale):
    if input.requires_grad:
        input.register_hook(lambda g: g * scale)


def exp(x, eps=1e-12):
    return (x < 0) * (x.clamp(max=0)).exp() + (x >= 0) / ((-x.clamp(min=0)).exp() + eps)


def log(x, eps=1e-12):
    return (x + eps).log()


def func(func_name):
    if func_name == 'tanh':
        return nn.Tanh()
    elif func_name == 'relu':
        return nn.ReLU()
    elif func_name == 'silu':
        return nn.SiLU()
    elif func_name == 'mish':
        return nn.Mish()
    elif func_name == 'sigmoid':
        return nn.Sigmoid()
    elif func_name == 'softmax':
        return nn.Softmax(dim=1)
    elif func_name == 'log_softmax':
        return nn.LogSoftmax(dim=1)
    else:
        assert False, "Invalid func_name."


class CheckBP(nn.Module):
    def __init__(self, label='a', show=1):
        super(CheckBP, self).__init__()
        self.label = label
        self.show = show

    def forward(self, input):
        return F.CheckBP.apply(input, self.label, self.show)


class Identity(nn.Module):
    def forward(self, input):
        return F.Identity.apply(input)
