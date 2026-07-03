import pandas as pd
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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from kmeans_pytorch import kmeans
import torch.nn.functional as F
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")

# SNARE_Mus_Cortex
# SNARE_Mus_Cortex_RNA
# 10xMultiome_PBMC10x  
# 10xMultiome_PBMC10x_RNA
# CITE_PBMC_Inhouse
# Tea_PBMC_PBMC
# Tea_PBMCratac
# Tea_PBMCradt
# Tea_PBMCr
# 10xMultiome_BMMC
# 10xMultiome_BMMC_RNA
# CITE_BMNC
# SHARE_Mus_Brain
# CITE _PBMC10x
# SHARE_Mus_Skin_filtered

Dataname = "SHARE_Mus_Skin_filtered" 
parser = argparse.ArgumentParser(description='train')
parser.add_argument("--omic1_dir", type=str, default='') # RNA dir
parser.add_argument("--omic2_dir", type=str, default='') # ATAC dir
parser.add_argument("--omic3_dir", type=str, default=None) # ADT dir
parser.add_argument("--omic_type", nargs='+', type=str, default=['rna', 'atac', 'adt'])
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--mse_epochs", default=500,) #stage1:pretrain AutoEncoders
parser.add_argument("--con_epochs", default=100)  #stage2:train all network
parser.add_argument("--temperature_l", default=1.0)
parser.add_argument("--lamda_P", default=1.0)
parser.add_argument("--lamda_Q", default=1.0)  #contrastive loss weight 1
parser.add_argument("--lamda_U", default=1.0)  #contrastive loss weight 2
parser.add_argument("--depth", default=5, type=int) # lambda_0
parser.add_argument("--noise", default=0.03, type=float)
parser.add_argument("--feature_dim_ls", nargs='+', type=int, default=[500, 60]) #RNA,ATAC,ADT latent dim
parser.add_argument("--reserve_dims", nargs='+', type=int, default=[7000, 6000]) #RNA,ATAC,ADT input dim
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

omic_dir_lst = []

if args.dataset == "SNARE_Mus_Cortex":
    args.omic1_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/SNARE_Mus_Cortex/RNA.h5ad'
    args.omic2_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/SNARE_Mus_Cortex/ATAC-Boruta.h5ad'
    args.learning_rate = 0.0003
    args.reserve_dims = [4000, 2000]
    args.omic_type = ['rna', 'atac']
    args.feature_dim_ls = [500, 500]
    args.mse_epochs = 200
    args.con_epochs = 50
    args.lamda_U = 0.5
    args.depth = 5
    args.noise = 0.003
    seed = 10
    omic_dir_lst = [args.omic1_dir, args.omic2_dir]
    
elif args.dataset == "SNARE_Mus_Cortex_RNA":
    args.omic1_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/SNARE_Mus_Cortex/RNA.h5ad'
    args.learning_rate = 0.0003
    args.reserve_dims = [4000]
    args.omic_type = ['rna']
    args.feature_dim_ls = [500]
    args.mse_epochs = 400
    args.con_epochs = 100
    args.lamda_U = 0.5
    args.depth = 5
    args.noise = 0.003
    seed = 10
    omic_dir_lst = [args.omic1_dir]
    
elif args.dataset == "10xMultiome_PBMC10x":
    args.omic1_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/10xMultiome_PBMC10x/10xMultiome_PBMC10x  -Multiome-Pbmc10k-RNA.h5ad'
    args.omic2_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/10xMultiome_PBMC10x/10xMultiome_PBMC10x  -Multiome-Pbmc10k-ATAC.h5ad'
    args.batch_size = 256
    args.learning_rate = 0.0003
    args.reserve_dims = [4000, 2500]
    args.omic_type = ['rna', 'atac']
    # args.feature_dim = 512
    args.feature_dim_ls = [500, 500]
    args.mse_epochs = 200
    args.con_epochs = 30
    args.lamda_U = 0.5
    args.depth = 5
    args.noise = 0.002
    seed = 10
    omic_dir_lst = [args.omic1_dir, args.omic2_dir]
    
elif args.dataset == "10xMultiome_PBMC10x_RNA":
    args.omic1_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/10xMultiome_PBMC10x  /10xMultiome_PBMC10x  -Multiome-Pbmc10k-RNA.h5ad'
    args.batch_size = 256
    args.learning_rate = 0.0003
    args.reserve_dims = [2000]
    args.omic_type = ['rna']
    args.feature_dim_ls = [512]
    args.mse_epochs = 300
    args.con_epochs = 150
    args.lamda_U = 0.5
    args.depth = 5
    args.noise = 0.002
    seed = 10
    omic_dir_lst = [args.omic1_dir]    

elif args.dataset == "CITE_PBMC_Inhouse":
    args.omic1_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/CITE_PBMC_Inhouse/data/rna.h5ad'
    args.omic2_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/CITE_PBMC_Inhouse/data/adt.h5ad'
    args.learning_rate = 0.0003
    args.omic_type = ['rna', 'adt']
    args.feature_dim_ls = [500, 500]
    args.reserve_dims = [4000, 2500]
    args.mse_epochs = 500
    args.con_epochs = 50
    args.lamda_U = 0.5
    args.depth = 5
    args.noise = 0.003
    seed = 10
    omic_dir_lst = [args.omic1_dir, args.omic2_dir]

elif args.dataset == "Tea_PBMC_PBMC":
    args.omic1_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/Matilda-data-Tea_PBMCseq/train_rna.h5ad'
    args.omic2_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/Matilda-data-Tea_PBMCseq/train_atac.h5ad'
    args.omic3_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/Matilda-data-Tea_PBMCseq/train_adt.h5ad'
    args.learning_rate = 0.0003
    args.reserve_dims = [3000, 2000, 46]
    args.omic_type = ['rna', 'atac', 'adt']
    args.feature_dim_ls = [512, 512, 256]
    args.mse_epochs = 20
    args.con_epochs = 60
    args.lamda_U = 0.5
    args.depth = 5
    args.noise = 0.003
    seed = 10
    omic_dir_lst = [args.omic1_dir, args.omic2_dir, args.omic3_dir]
    
elif args.dataset == "Tea_PBMCratac":
    args.omic1_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/Matilda-data-Tea_PBMCseq/train_rna.h5ad'
    args.omic2_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/Matilda-data-Tea_PBMCseq/train_atac.h5ad'
    args.learning_rate = 0.0003
    args.reserve_dims = [3000, 2000]
    args.omic_type = ['rna', 'atac']
    args.feature_dim_ls = [512, 512]
    args.mse_epochs = 50
    args.con_epochs = 200
    args.lamda_U = 0.5
    args.depth = 5
    args.noise = 0.003
    seed = 10
    omic_dir_lst = [args.omic1_dir, args.omic2_dir]    
    
elif args.dataset == "Tea_PBMCradt":
    args.omic1_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/Matilda-data-Tea_PBMCseq/train_rna.h5ad'
    args.omic2_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/Matilda-data-Tea_PBMCseq/train_adt.h5ad'
    args.learning_rate = 0.0003
    args.reserve_dims = [3000, 46]
    args.omic_type = ['rna', 'adt']
    args.feature_dim_ls = [512, 256]
    args.mse_epochs = 50
    args.con_epochs = 200
    args.lamda_U = 0.5
    args.depth = 5
    args.noise = 0.003
    seed = 10
    omic_dir_lst = [args.omic1_dir, args.omic2_dir]     
    
elif args.dataset == "Tea_PBMCr":
    args.omic1_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/Matilda-data-Tea_PBMCseq/train_rna.h5ad'
    args.learning_rate = 0.0003
    args.reserve_dims = [3000]
    args.omic_type = ['rna']
    args.feature_dim_ls = [512]
    args.mse_epochs = 50
    args.con_epochs = 100
    args.lamda_U = 0.5
    args.depth = 4
    args.noise = 0.003
    seed = 10
    omic_dir_lst = [args.omic1_dir]        

elif args.dataset == '10xMultiome_BMMC':
    args.omic1_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/10xMultiome _BMMC/rna.h5ad'
    args.omic2_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/10xMultiome _BMMC/atac.h5ad'
    args.learning_rate = 0.0003
    args.reserve_dims = [3000, 1000]
    args.omic_type = ['rna', 'atac']
    args.feature_dim_ls = [512, 512]
    args.mse_epochs = 90
    args.con_epochs = 200
    args.lamda_U = 0.5
    args.depth = 5
    args.noise = 0.003
    seed = 10
    omic_dir_lst = [args.omic1_dir, args.omic2_dir]
    
elif args.dataset == '10xMultiome_BMMC_RNA':
    args.omic1_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/10xMultiome _BMMC/rna.h5ad'
    args.learning_rate = 0.0003
    args.reserve_dims = [3000]
    args.omic_type = ['rna']
    args.feature_dim_ls = [512]
    args.mse_epochs = 400
    args.con_epochs = 200
    args.lamda_U = 0.5
    args.depth = 5
    args.noise = 0.003
    seed = 10
    omic_dir_lst = [args.omic1_dir]
    
elif args.dataset == 'CITE_BMNC':
    args.omic1_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/CITE_BMNC/rna.h5ad'
    args.omic2_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/CITE_BMNC/adt.h5ad'
    args.batch_size = 256
    args.learning_rate = 0.0003
    args.reserve_dims = [2000, 25]
    args.omic_type = ['rna', 'adt']
    args.feature_dim_ls = [500, 300]
    args.mse_epochs = 30
    args.con_epochs = 250
    args.lamda_U = 0.5
    args.depth = 5
    args.noise = 0.002
    seed = 10
    omic_dir_lst = [args.omic1_dir, args.omic2_dir]

elif args.dataset == 'SHARE_Mus_Brain':
    args.omic1_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/SHARE_Mus_Brain/rna.h5ad'
    args.omic2_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/SHARE_Mus_Brain/atac.h5ad'
    args.batch_size = 256
    args.learning_rate = 0.0003
    args.reserve_dims = [3000, 1000]
    args.omic_type = ['rna', 'atac']
    args.feature_dim_ls = [512, 128]
    args.mse_epochs = 100
    args.con_epochs = 200
    args.lamda_Q = 0.001
    args.lamda_U = 0.001
    args.depth = 3
    args.noise = 0.002
    seed = 10
    omic_dir_lst = [args.omic1_dir, args.omic2_dir] #
    
elif args.dataset == 'CITE_PBMC10x':
    args.omic1_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/CITE _PBMC10x/rna.h5ad'
    args.omic2_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/CITE _PBMC10x/adt.h5ad'
    args.batch_size = 256
    args.learning_rate = 0.0003
    args.reserve_dims = [500, 10]
    args.omic_type = ['rna', 'adt']
    args.feature_dim_ls = [400, 50]
    args.mse_epochs = 30
    args.con_epochs = 100
    args.lamda_U = 0.5
    args.depth = 5
    args.noise = 0.003
    seed = 10
    omic_dir_lst = [args.omic1_dir, args.omic2_dir]  
    
elif args.dataset == 'SHARE_Mus_Skin_filtered':
    args.omic1_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/SHARE_Mus_Skin_filtered/rna.h5ad'
    args.omic2_dir = '/home/zqzhao/workplace/Multi_omics_unet/dataset/SHARE_Mus_Skin_filtered/atac.h5ad'
    args.batch_size = 256
    args.learning_rate = 0.0003
    args.reserve_dims = [3000, 3000] #[1000, 25]
    args.omic_type = ['rna', 'atac']
    args.feature_dim_ls = [500, 500]
    args.mse_epochs = 500 
    args.con_epochs = 120 
    args.lamda_Q = 0.001
    args.lamda_U = 0.0
    args.depth = 6
    args.noise = 0.0002
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

def init_kmeans_plus(cluster_temp, k):
    n = cluster_temp.shape[0]
    if k > n or k <= 0:
        raise ValueError("k value invalid")

    is_selected = torch.zeros(n, dtype=torch.bool)
    selected_indices = torch.zeros(k, dtype=torch.long)

    i = torch.randint(n, (1,)).item()
    is_selected[i] = True
    selected_indices[0] = i

    for i in range(1, k):
        candidate_indices = torch.nonzero(~is_selected).squeeze()
        if candidate_indices.numel() == 0:
            break

        cluster_selected = cluster_temp[selected_indices[:i]]
        cluster_candidates = cluster_temp[candidate_indices]

        dists = torch.cdist(cluster_candidates, cluster_selected)

        min_dists = dists.sum(dim=1)[0]

        selected_index = candidate_indices[torch.argmax(min_dists)]
        selected_indices[i] = selected_index
        is_selected[selected_index] = True

    return cluster_temp[selected_indices]     
    
def pretrain(epoch):
    tot_loss = 0.
    
    for batch_idx, (xs, _) in enumerate(full_data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)

        optimizer.zero_grad()
        xrs = model(xs, True)
        loss_list = []
        for v in range(view):
            loss_list.append(F.mse_loss(xrs[v], xs[v], reduction='mean'))

        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(full_data_loader)))
    return tot_loss / len(full_data_loader)


def contrastive_train(epoch,lamda_P,lamda_Q, lamda_U):
    tot_loss = 0.
    for batch_idx, (xs, _) in enumerate(clean_data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
            
        optimizer.zero_grad()
        xrs, P, Qs, Qs_drop, cls, glb_feature = model(xs)
        
        loss_list = []
        for v in range(view):
            for w in range(v+1, view):
                loss_list.append(lamda_Q * criterion1.forward_label(Qs[v], Qs[w]))
                loss_list.append(lamda_U * criterion1.forward_label(Qs[v], Qs_drop[w]))
                loss_list.append(lamda_U * criterion1.forward_label(Qs_drop[v], Qs[w]))
            loss_list.append(lamda_P * F.kl_div(torch.log(P), Qs[v]))
            loss_list.append(F.mse_loss(xs[v], xrs[v], reduction='mean'))
        
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
        
        model.copy_weight()
        
        
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(clean_data_loader)))
    return tot_loss/len(clean_data_loader)

if __name__=="__main__":
    if not os.path.exists('./models'):
        os.makedirs('./models')
    T = 1
    for i in range(T):
        setup_seed(seed)
        full_dataset, dims, view, data_size, class_num = load_data(omic_dir_lst, args.omic_type, args.reserve_dims)
        full_data_loader = torch.utils.data.DataLoader(
            full_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
        )
        max_res = [0.0, 0]
        print("ROUND:{} DataName:{} view_num:{}".format(i + 1, Dataname, view))
        model = Network(view, dims, args.feature_dim_ls, class_num, args.depth, args.noise, device)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        criterion1 = ContrastiveWithEntropyLoss(args.batch_size, class_num, args.temperature_l, device).to(device)
        
        epoch = 1
        rec_loss_ls = []
        while epoch <= args.mse_epochs:
            rec_loss_ls.append(pretrain(epoch))
            epoch += 1
        ############################################## Convergence analysis
        folder = "result/"
        if not os.path.exists(folder):
            os.mkdir(folder)
        plot_vis_losses(rec_loss_ls)
        torch.save(model.state_dict(), './models/pre_' + args.dataset + '.pth')
        # pd.Series(rec_loss_ls, index=[i for i in range(1,epoch)], name='pretrain_loss').to_csv(f'/home/zqzhao/workplace/scEDCA/result/{Dataname}_pretrain_loss.csv')
        ############################################## init As
        select_hidden_lqc_ls = []
        with torch.no_grad():
            
            raw_unpaired_clean_data = full_dataset.unpaired_clean_data()
            Zs = []
            for v in range(view):
                raw_unpaired_clean_data[v] = raw_unpaired_clean_data[v].to(device)
            for v in range(view):
                hidden = model.encoders[v](raw_unpaired_clean_data[v])
                cluster_temp = hidden.detach().cpu()
                try:
                    hidden_hqc_idx = filter_noise_latent_cells(cluster_temp.numpy())
                    # select_hidden_lqc_ls.append()
                    cluster_temp = cluster_temp[hidden_hqc_idx]
                except:
                    pass
                
                # if shift center is not NAN, use this snap code
                # cluster_ids_x, cluster_centers = kmeans(X=cluster_temp, num_clusters=class_num, distance='cosine', device=device)
                # model.As[v].data = torch.tensor(cluster_centers).to(device)

                # if shift center is NAN, use this snap code
                if args.omic_type[v] != 'rna':
                    cluster_centers = init_kmeans_plus(cluster_temp, class_num)
                    model.As[v].data = torch.tensor(cluster_centers).to(device)
                else:
                    cluster_ids_x, cluster_centers = kmeans(X=cluster_temp, num_clusters=class_num, distance='cosine', device=device)
                    model.As[v].data = torch.tensor(cluster_centers).to(device)

        # if filter sample size >= k, use this snap code
        # raw_paired_clean_data, raw_clean_labels = full_dataset.paired_clean_data()

        # if filter sample size < k, use this snap code
        raw_paired_clean_data, raw_clean_labels = full_dataset.full_data()  #CITE _PBMC_Inhouse

        raw_paired_clean_data = [u.numpy() for u in raw_paired_clean_data]
        clean_dataset = MultiOmicsCleanDataset(raw_paired_clean_data, raw_clean_labels)
        clean_data_loader = torch.utils.data.DataLoader(
            clean_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
        )

        rec_loss_ls = []
        acc_ls = []
        while epoch <= args.mse_epochs + args.con_epochs:
            model.train()
            loss = contrastive_train(epoch,args.lamda_P,args.lamda_Q,args.lamda_U)
            rec_loss_ls.append(loss)
            acc = valid(model, device, full_dataset, view, data_size, isprint=False, return_latent=False, return_full_metric=False)
            acc_ls.append(acc)
            if acc > max_res[0]:
                max_res = [acc, epoch - args.mse_epochs]
                state = model.state_dict()
                torch.save(state, './models/' + args.dataset + '.pth')
            if epoch == args.mse_epochs + args.con_epochs:
                print('--------args----------')
                for k in list(vars(args).keys()):
                    print('%s: %s' % (k, vars(args)[k]))
                print('--------args----------')
                checkpoint = torch.load('./models/' + args.dataset + '.pth')
                model.load_state_dict(checkpoint)
                print('Full dataset:')
                acc, nmi, pur, ari, pca_casw, pca_clisi, lda_casw, lda_clisi, target_pred, glb_vector = valid(model, device, full_dataset, view, data_size, isprint=True, return_latent=True, return_full_metric=True)
                # save_latent(glb_vector, Dataname)
                np.save(f'./latent/{args.dataset}_glb_vector.npy', glb_vector)
                np.save(f'./latent/{args.dataset}_target_pred.npy', target_pred)
                # print('Clean dataset:')
                # valid(model, device, clean_dataset, view, data_size, isprint=True)
            epoch += 1
        # pd.Series(rec_loss_ls, index=[i for i in range(args.mse_epochs+1,epoch)], name='train_loss').to_csv(f'/home/zqzhao/workplace/scEDCA/result/{Dataname}_train_loss.csv')
        # pd.Series(acc_ls, index=[i for i in range(args.mse_epochs+1,epoch)], name='train_acc').to_csv(f'/home/zqzhao/workplace/scEDCA/result/{Dataname}_train_acc.csv')
        plot_loss_and_accuracy(rec_loss_ls, acc_ls, f_name=Dataname)
        

 
