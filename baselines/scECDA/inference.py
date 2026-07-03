import math
import torch
from network3 import Network
from metric import valid
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from loss import ContrastiveWithEntropyLoss
from utils import plot_vis_losses, plot_loss_and_accuracy, save_latent
from dataloader_hh import load_data, MultiOmicsCleanDataset, filter_noise_latent_cells
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from sklearn.cluster import KMeans
from kmeans_pytorch import kmeans
import torch.nn.functional as F
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")

# chen
# 10x
# inhouse
# tea-seq
# tearatac
# tearadt
# BMMC
# BMNC
# Mouse_Brain
# pbmc10x

Dataname = "BMMC" 
parser = argparse.ArgumentParser(description='train')
parser.add_argument("--omic1_dir", type=str, default='')
parser.add_argument("--omic2_dir", type=str, default='')
parser.add_argument("--omic3_dir", type=str, default=None)
parser.add_argument("--omic_type", nargs='+', type=str, default=['rna', 'atac', 'adt'])
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--mse_epochs", default=1200)
parser.add_argument("--con_epochs", default=100)
parser.add_argument("--temperature_l", default=1.0)
parser.add_argument("--lamda_P", default=1.0)
parser.add_argument("--lamda_Q", default=1.0)
parser.add_argument("--lamda_U", default=1.0)
parser.add_argument("--depth", default=5, type=int)
parser.add_argument("--noise", default=0.03, type=float)
parser.add_argument("--feature_dim_ls", nargs='+', type=int, default=[500, 60])
parser.add_argument("--reserve_dims", nargs='+', type=int, default=[7000, 6000])
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

omic_dir_lst = []

if args.dataset == "chen":
    args.omic1_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/Chen/RNA.h5ad'
    args.omic2_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/Chen/ATAC-Boruta.h5ad'
    args.learning_rate = 0.0003
    args.reserve_dims = [4000, 2000] #[4000, 2000]
    args.omic_type = ['rna', 'atac']
    args.feature_dim_ls = [500, 500]
    # args.feature_dim = 512
    args.mse_epochs = 200 #200
    args.con_epochs = 50 #25
    args.lamda_U = 0.5
    args.depth = 5
    args.noise = 0.003
    seed = 10
    omic_dir_lst = [args.omic1_dir, args.omic2_dir]
    
elif args.dataset == "10x":
    args.omic1_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/10x/10x-Multiome-Pbmc10k-RNA.h5ad'
    args.omic2_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/10x/10x-Multiome-Pbmc10k-ATAC.h5ad'
    args.batch_size = 256
    args.learning_rate = 0.0003 #0.0002
    args.reserve_dims = [4000, 2500] #[4000, 3000]
    args.omic_type = ['rna', 'atac']
    # args.feature_dim = 512
    args.feature_dim_ls = [500, 500]
    args.mse_epochs = 200 #840
    args.con_epochs = 30 #60
    args.lamda_U = 0.5 #0.
    args.depth = 5
    args.noise = 0.002
    seed = 10
    omic_dir_lst = [args.omic1_dir, args.omic2_dir]

elif args.dataset == "inhouse":
    args.omic1_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/INhouse/data/rna.h5ad'
    args.omic2_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/INhouse/data/adt.h5ad'
    args.learning_rate = 0.0003
    args.omic_type = ['rna', 'adt']
    args.feature_dim_ls = [500, 500]
    args.reserve_dims = [4000, 2500]
    args.mse_epochs = 500 #800
    args.con_epochs = 50 #25  #10
    args.lamda_U = 0.5 #1.
    args.depth = 5
    args.noise = 0.003
    seed = 10
    omic_dir_lst = [args.omic1_dir, args.omic2_dir]
    
elif args.dataset == "Ma53":
    args.omic1_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/Ma/rna_batch53.h5ad'
    args.omic2_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/Ma/atac_boruta_batch53.h5ad'
    args.batch_size = 512
    args.reserve_dims = [4000, 2500] #[7000, 3000]
    args.omic_type = ['rna', 'atac']
    args.learning_rate = 0.0003
    args.feature_dim_ls = [500, 500]
    args.mse_epochs = 200 
    args.con_epochs = 50 #25  #10
    args.lamda_U = 0.5
    args.depth = 5
    args.noise = 0.003
    seed = 10 
    omic_dir_lst = [args.omic1_dir, args.omic2_dir]

elif args.dataset == "Ma54":
    args.omic1_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/Ma/rna_batch54.h5ad'
    args.omic2_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/Ma/atac_boruta_batch54.h5ad'
    args.batch_size = 512
    args.learning_rate = 0.0003
    args.omic_type = ['rna', 'atac']
    args.feature_dim = 512
    args.mse_epochs = 700 #840
    args.con_epochs = 25 #25  #10
    args.lamda_U = 1.
    args.depth = 5
    args.noise = 0.003
    seed = 10 
    omic_dir_lst = [args.omic1_dir, args.omic2_dir]

elif args.dataset == "Ma55":
    args.omic1_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/Ma/rna_batch55.h5ad'
    args.omic2_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/Ma/atac_boruta_batch55.h5ad'
    args.learning_rate = 0.0003
    args.omic_type = ['rna', 'atac']
    args.feature_dim = 512
    args.mse_epochs = 800 #840
    args.con_epochs = 35 #25  #10
    args.lamda_U = 1.
    args.depth = 6
    args.noise = 0.003
    seed = 10 
    omic_dir_lst = [args.omic1_dir, args.omic2_dir]

elif args.dataset == "Ma56":
    args.omic1_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/Ma/rna_batch56.h5ad'
    args.omic2_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/Ma/atac_boruta_batch56.h5ad'
    args.learning_rate = 0.0003
    args.omic_type = ['rna', 'atac']
    args.feature_dim = 512
    args.mse_epochs = 800 #840
    args.con_epochs = 35 #25  #10
    args.lamda_U = 1.
    args.depth = 6
    args.noise = 0.003
    seed = 10
    omic_dir_lst = [args.omic1_dir, args.omic2_dir]

elif args.dataset == "Ma_split60":
    args.omic1_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/Ma/split60_rna_train.h5ad'
    args.omic2_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/Ma/split60_atac_train.h5ad'
    args.learning_rate = 0.0003
    args.omic_type = ['rna', 'atac']
    args.feature_dim = 512
    args.mse_epochs = 800 #840
    args.con_epochs = 35 #25  #10
    args.lamda_U = 1.
    args.depth = 6
    args.noise = 0.003
    seed = 10
    omic_dir_lst = [args.omic1_dir, args.omic2_dir]
    
elif args.dataset == "Ma_all":
    args.omic1_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/Ma/Ma-2020-RNA.h5ad'
    args.omic2_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/Ma/Ma-2020-ATAC.h5ad'
    args.learning_rate = 0.0003
    args.omic_type = ['rna', 'atac']
    args.feature_dim = 512
    args.mse_epochs = 800 #840
    args.con_epochs = 35 #25  #10
    args.lamda_U = 1.
    args.depth = 6
    args.noise = 0.003
    seed = 10
    omic_dir_lst = [args.omic1_dir, args.omic2_dir]

elif args.dataset == "tea-seq":
    args.omic1_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/Matilda-data-TEAseq/train_rna.h5ad'
    args.omic2_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/Matilda-data-TEAseq/train_atac.h5ad'
    args.omic3_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/Matilda-data-TEAseq/train_adt.h5ad'
    args.learning_rate = 0.0003
    args.reserve_dims = [3000, 2000, 46] #[7000, 6000, 6000]
    args.omic_type = ['rna', 'atac', 'adt']
    args.feature_dim_ls = [512, 512, 256]
    args.mse_epochs = 20 #20 #800
    args.con_epochs = 60 #35 
    args.lamda_U = 0.5 #
    args.depth = 5
    args.noise = 0.003
    seed = 10
    omic_dir_lst = [args.omic1_dir, args.omic2_dir, args.omic3_dir]
    
elif args.dataset == "tearatac":
    args.omic1_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/Matilda-data-TEAseq/train_rna.h5ad'
    args.omic2_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/Matilda-data-TEAseq/train_atac.h5ad'
    args.learning_rate = 0.0003
    args.reserve_dims = [3000, 2000] #[7000, 6000, 6000]
    args.omic_type = ['rna', 'atac']
    args.feature_dim_ls = [512, 512]
    args.mse_epochs = 20 #20 #800
    args.con_epochs = 60 #35 
    args.lamda_U = 0.5 #
    args.depth = 5
    args.noise = 0.003
    seed = 10
    omic_dir_lst = [args.omic1_dir, args.omic2_dir]    
    
elif args.dataset == "tearadt":
    args.omic1_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/Matilda-data-TEAseq/train_rna.h5ad'
    args.omic2_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/Matilda-data-TEAseq/train_adt.h5ad'
    args.learning_rate = 0.0003
    args.reserve_dims = [3000, 46] #[7000, 6000, 6000]
    args.omic_type = ['rna', 'adt']
    args.feature_dim_ls = [512, 256]
    args.mse_epochs = 10 #20 #800
    args.con_epochs = 100 #35 
    args.lamda_U = 0.5 #
    args.depth = 5
    args.noise = 0.003
    seed = 10
    omic_dir_lst = [args.omic1_dir, args.omic2_dir]        

elif args.dataset == 'BMMC':
    args.omic1_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/BMMC/rna.h5ad'
    args.omic2_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/BMMC/atac.h5ad'
    args.learning_rate = 0.0003
    args.reserve_dims = [3000, 1000] #[4000, 2000] #[7000, 6000]
    args.omic_type = ['rna', 'atac']
    args.feature_dim_ls = [512, 512]
    args.mse_epochs = 90 #80 #200
    args.con_epochs = 200 #160  #10
    args.lamda_U = 0.5
    args.depth = 5
    args.noise = 0.003
    seed = 10
    omic_dir_lst = [args.omic1_dir, args.omic2_dir]
    
elif args.dataset == 'BMNC':
    args.omic1_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/BMNC/rna.h5ad'
    args.omic2_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/BMNC/adt.h5ad'
    args.batch_size = 256
    args.learning_rate = 0.0003
    args.reserve_dims = [2000, 25] #[1000, 25]
    args.omic_type = ['rna', 'adt']
    # args.feature_dim = 320 # 512
    args.feature_dim_ls = [500, 300]
    args.mse_epochs = 30 #30
    args.con_epochs = 250 #10
    args.lamda_U = 0.5
    args.depth = 5
    args.noise = 0.002
    seed = 10
    omic_dir_lst = [args.omic1_dir, args.omic2_dir]    

elif args.dataset == 'Mouse_Brain':
    args.omic1_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/Mouse_Brain/rna.h5ad'
    args.omic2_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/Mouse_Brain/atac.h5ad'
    args.batch_size = 256
    args.learning_rate = 0.0003
    args.reserve_dims = [4000] #[4000, 500]
    args.omic_type = ['rna'] #['rna', 'atac']
    args.feature_dim_ls = [500] #[500, 50]
    args.mse_epochs = 400 #840
    args.con_epochs = 50 #25  #10
    args.lamda_U = 0.5
    args.depth = 5
    args.noise = 0.002
    seed = 10
    omic_dir_lst = [args.omic1_dir, ] #args.omic2_dir
    
elif args.dataset == 'pbmc10x':
    args.omic1_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/pbmc10x/rna.h5ad'
    args.omic2_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/pbmc10x/adt.h5ad'
    args.batch_size = 256
    args.learning_rate = 0.0003
    args.reserve_dims = [500, 10] #[500, 10]
    args.omic_type = ['rna', 'adt']
    args.feature_dim_ls = [400, 50] #400
    args.mse_epochs = 30 #90 #400
    args.con_epochs = 100#35 #25  #10
    args.lamda_U = 0.5 #1.
    args.depth = 5
    args.noise = 0.003
    seed = 10
    omic_dir_lst = [args.omic1_dir, args.omic2_dir]  

def setup_seed(seed):
   random.seed(seed)
   os.environ["PYTHONHASHSEED"] = str(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True


if __name__=="__main__":
    setup_seed(seed)
    full_dataset, dims, view, data_size, class_num = load_data(omic_dir_lst, args.omic_type, args.reserve_dims)
    
    print("DataName:{} view_num:{}".format(Dataname, view))
    model = Network(view, dims, args.feature_dim_ls, class_num, args.depth, args.noise, device)
    model = model.to(device)
    checkpoint = torch.load(f'./models/{args.dataset} best.pth')
    model.load_state_dict(checkpoint)
    
    acc, nmi, pur, ari, target_pred, glb_vector = valid(model, device, full_dataset, view, data_size, isprint=True, return_latent=True)
    # save_latent(glb_vector, Dataname)
    np.save(f'./latent/{args.dataset}_glb_vector.npy', glb_vector)
    np.save(f'./latent/{args.dataset}_target_pred.npy', target_pred)