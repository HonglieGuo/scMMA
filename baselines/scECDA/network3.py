import torch.nn as nn
from torch.nn.functional import normalize
import torch.nn.functional as F
import torch
from diffFormer import DiffFormerEncoderLayer, weights_init


import torch.nn as nn

# 定义共享的编码器模块
class SharedEncoder(nn.Module):
    def __init__(self, input_dim, feature_dim, dropout_rate=0.25):
        super(SharedEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.Mish(),
            nn.Dropout(dropout_rate),  # Dropout layer added
            nn.Linear(500, 500),
            nn.Mish(),
            nn.Dropout(dropout_rate),  # Dropout layer added
            nn.Linear(500, 2000),
            nn.Mish(),
            nn.Dropout(dropout_rate),  # Dropout layer added
            nn.Linear(2000, feature_dim),
        )
        
    def forward(self, x):
        return self.encoder(x)

# Encoder1 使用共享编码器
class Encoder1(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder1, self).__init__()
        self.shared_encoder = SharedEncoder(input_dim, feature_dim, dropout_rate=0.0)

    def forward(self, x):
        return self.shared_encoder(x)

# Encoder2 使用共享编码器
class Encoder2(nn.Module):
    def __init__(self, input_dim, feature_dim, noise, dropout_rate=0.25):
        super(Encoder2, self).__init__()
        self.noise = noise
        self.shared_encoder = SharedEncoder(input_dim, feature_dim, dropout_rate=dropout_rate)
        self.setParamGradFalse()

    def setParamGradFalse(self):
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        x = self.shared_encoder(x)
        x = x + torch.randn_like(x) * self.noise
        return x



class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.Mish(),
            nn.Linear(2000, 500),
            nn.Mish(),
            nn.Linear(500, 500),
            nn.Mish(),
            nn.Linear(500, input_dim),
        )

    def forward(self, x):
        return self.decoder(x)


# view-cross fusion
class VCF(nn.Module):
    def __init__(self,in_feature_dim,class_num,depth):
        super(VCF,self).__init__()
        # TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=in_feature_dim, nhead=1,dim_feedforward=512)
        # self.TransformerEncoder = nn.TransformerEncoder(TransformerEncoderLayer, num_layers=1)
        self.TransformerEncoder = DiffFormerEncoderLayer(in_feature_dim, 1, depth)
        self.cluster = nn.Sequential(
            nn.Linear(in_feature_dim, class_num),
            nn.Softmax(dim=1),
        )
    def forward(self,C):
        if len(C.shape) <= 2:
            C = C.unsqueeze(0)
            sample_feature = self.TransformerEncoder(C)
            sample_feature = sample_feature.squeeze(0)
        else:
            sample_feature = self.TransformerEncoder(C)
        cls = self.cluster(sample_feature)
        return cls, sample_feature

class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim_ls, class_num, depth, noise, device):
        super(Network, self).__init__()
        self.view = view
        self.encoders = []
        self.encoders_drop = []
        self.decoders = []
        self.As = []
        for v in range(view):
            self.encoders.append(Encoder1(input_size[v], feature_dim_ls[v]).to(device))
            self.encoders_drop.append(Encoder2(input_size[v], feature_dim_ls[v], noise).to(device))
            self.decoders.append(Decoder(input_size[v], feature_dim_ls[v]).to(device))
            self.As.append(nn.Parameter(torch.Tensor(class_num, feature_dim_ls[v])).to(device))  # 质心
            torch.nn.init.xavier_normal_(self.As[v].data)
            # torch.nn.init.kaiming_normal_(self.As[v].data)

        self.encoders = nn.ModuleList(self.encoders)
        self.encoders_drop = nn.ModuleList(self.encoders_drop)
        self.decoders = nn.ModuleList(self.decoders)
        # self.As = nn.ParameterList(self.As)
        self.VCF = VCF(sum(feature_dim_ls), class_num, depth)
        self.alpha = 1.0

    def forward(self,X,preTrain=False):
        Zs = []
        transform_Zs = []
        X_hat = []
        for v in range(self.view):
            Z = self.encoders[v](X[v])
            # Z_drop = self.encoders_drop[v](X[v])
            if not preTrain:
                Zs.append(Z)
            # transform_Zs.append(Z_drop)
            X_hat.append(self.decoders[v](Z))
        if preTrain:
            return X_hat

        for v in range(self.view):
            Z_drop = self.encoders_drop[v](X[v])
            transform_Zs.append(Z_drop)
        
        Z = torch.cat(Zs, dim=1)
        
        cls, glb_feature = self.VCF(Z)
        P = self.target_distribution(cls)
        Qs = []
        Qs_drop = []
        for v in range(self.view):
            q = 1.0 / (1.0 + torch.sum(torch.pow(Zs[v].unsqueeze(1) - self.As[v], 2), 2) / self.alpha)
            q = q.pow((self.alpha + 1.0) / 2.0)
            q = (q.t() / torch.sum(q, 1)).t()
            Qs.append(q)
            
            q = 1.0 / (1.0 + torch.sum(torch.pow(transform_Zs[v].unsqueeze(1) - self.As[v], 2), 2) / self.alpha)
            q = q.pow((self.alpha + 1.0) / 2.0)
            q = (q.t() / torch.sum(q, 1)).t()
            Qs_drop.append(q)
        return X_hat, P, Qs, Qs_drop, cls, glb_feature

    def forward_plot(self,X):
        Zs = []
        for v in range(self.view):
            Z = self.encoders[v](X[v])
            Zs.append(Z)
        Z = torch.cat(Zs, dim=1)
        t, fusioned_var = self.VCF(Z)
        P = self.target_distribution(t)
        preds = torch.argmax(P, dim=1)

        return Z, fusioned_var, preds

    def target_distribution(self,p):
        weight = p ** 2 / p.sum(0)
        return (weight.t() / weight.sum(1)).t()
    
    def copy_weight(self):
        for v in range(self.view):
            self.encoders_drop[v].load_state_dict(self.encoders[v].state_dict())
    
