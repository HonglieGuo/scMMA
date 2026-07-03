import os
import sys
import time
import argparse
import json
import h5py
import numpy as np
import scipy.sparse as sp
import torch as th
import torch.nn as nn
import anndata as ad
import scanpy as sc

# Append MINERVA subdirectory to Python path so we can import its modules
sys.path.append(os.path.abspath('./MINERVA'))
from modules import models, utils

# Set random seeds
seed = 1234
np.random.seed(seed)
th.manual_seed(seed)
if th.cuda.is_available():
    th.cuda.manual_seed_all(seed)

def load_h5(path):
    with h5py.File(path, "r") as f:
        X = np.array(f['matrix/data']).transpose()
        features = np.array(f['matrix/features']).astype(str)
        barcodes = np.array(f['matrix/barcodes']).astype(str)
    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.var_names = features
    adata.obs_names = barcodes
    return adata

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path1", nargs="+", required=True, help="Path to RNA h5 (can be multiple)")
    parser.add_argument("--path2", nargs="+", required=True, help="Path to ADT h5 (can be multiple)")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save output")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs to train (restored to official default)")
    args = parser.parse_args()
    
    start_time = time.time()
    
    # 1. Load data
    print("Loading data...")
    rna_adatas = []
    for i, p in enumerate(args.path1):
        adata = load_h5(p)
        adata.obs["batch"] = i
        rna_adatas.append(adata)
    rna = ad.concat(rna_adatas, join="inner")
    
    adt_adatas = []
    for i, p in enumerate(args.path2):
        adata = load_h5(p)
        adata.obs["batch"] = i
        adt_adatas.append(adata)
    adt = ad.concat(adt_adatas, join="inner")
    
    # Ensure features align and cells align
    assert rna.shape[0] == adt.shape[0], "RNA and ADT cell counts mismatch!"
    
    n_genes = rna.shape[1]
    n_proteins = adt.shape[1]
    n_cells = rna.shape[0]
    n_batches = len(args.path1)
    
    print(f"Data shape: {n_cells} cells, {n_genes} genes, {n_proteins} proteins, {n_batches} batches.")
    
    # Convert expression matrices to dense arrays for PyTorch training (MINERVA logic expects dense floats)
    rna_data = rna.X.toarray().astype(np.float32)
    adt_data = adt.X.toarray().astype(np.float32)
    batch_ids = rna.obs["batch"].values.astype(np.int64)
    
    # 2. Setup MINERVA namespace 'o'
    o = argparse.Namespace()
    o.task = "baseline"
    o.reference = ""
    o.experiment = "e0"
    o.rf_experiment = ""
    o.model = "default"
    o.actions = ["train"]
    o.method = "midas"
    o.init_model = ""
    o.init_from_ref = 0
    o.sample_num = 0
    o.input_mods = []
    o.train_ratio = 1.0 # use all cells for joint representation learning
    o.epoch_num = args.epochs
    o.batch_size = 128
    o.lr = 1e-4
    o.grad_clip = -1
    o.s_drop_rate = 0.1
    o.seed = seed
    o.use_shm = 0
    o.print_iters = -1
    o.log_epochs = 100
    o.save_epochs = 100
    o.time = 0
    o.debug = 0
    o.pretext = ["raw"]
    o.mask_ratio = 0.3
    
    # Hardcoded default model configurations from model.toml
    o.dim_c = 32
    o.dims_enc_s = [64, 64]
    o.dims_enc_chr = [128, 32]
    o.dims_enc_x = [256, 128]
    o.dims_discriminator = [128, 64]
    o.norm = "ln"
    o.drop = 0.2
    
    # Set data configuration
    o.mods = ["rna", "adt"]
    o.ref_mods = ["rna", "adt"]
    o.dims_x = {"rna": n_genes, "adt": n_proteins}
    o.mod_num = 2
    
    # Set up batch configuration
    o.s_joint = [[b] for b in range(n_batches)]
    o.combs = [[["rna", "adt"]] for _ in range(n_batches)]
    o.s_joint, o.combs, o.s, o.dims_s = utils.gen_all_batch_ids(o.s_joint, o.combs)
    o.dim_s = o.dims_s["joint"]
    o.dim_b = 2
    o.dim_z = o.dim_c + o.dim_b
    o.dims_dec_x = o.dims_enc_x[::-1]
    o.dims_dec_s = o.dims_enc_s[::-1]
    o.dims_h = {"rna": n_genes, "adt": n_proteins}
    o.ref_epoch_num = 0
    
    # 3. Initialize models and optimizers
    net = models.Net(o).cuda()
    discriminator = models.Discriminator(o).cuda()
    
    optimizer_net = th.optim.AdamW(net.parameters(), lr=o.lr)
    optimizer_disc = th.optim.AdamW(discriminator.parameters(), lr=o.lr)
    
    # 4. Training loop
    print("Training MINERVA...")
    dataset_indices = np.arange(n_cells)
    
    from tqdm import tqdm
    pbar = tqdm(range(args.epochs), desc="MINERVA Training", unit="epoch")
    for epoch in pbar:
        net.train()
        discriminator.train()
        
        np.random.shuffle(dataset_indices)
        epoch_loss = 0.0
        n_batches_in_epoch = int(np.ceil(n_cells / o.batch_size))
        
        for b_idx in range(n_batches_in_epoch):
            indices = dataset_indices[b_idx * o.batch_size : (b_idx + 1) * o.batch_size]
            if len(indices) < 2:
                continue
            
            # Format inputs dictionary
            inputs = {
                "raw": {
                    "rna": th.tensor(rna_data[indices]).cuda(),
                    "adt": th.tensor(adt_data[indices]).cuda(),
                },
                "s": {
                    "joint": th.tensor(batch_ids[indices]).unsqueeze(1).cuda(),
                    "rna": th.tensor(batch_ids[indices]).unsqueeze(1).cuda(),
                    "adt": th.tensor(batch_ids[indices]).unsqueeze(1).cuda(),
                },
                "e": {
                    "rna": th.ones(len(indices), n_genes).cuda(),
                    "adt": th.ones(len(indices), n_proteins).cuda(),
                }
            }
            
            # Net forward
            sum_losses, loss_net, raw_loss, c_all = net(inputs)
            
            # Discriminator forward & backward
            discriminator.epoch = epoch
            loss_disc = discriminator(utils.detach_tensors(c_all), inputs["s"])
            optimizer_disc.zero_grad()
            loss_disc.backward()
            optimizer_disc.step()
            
            # Generator forward & backward (net training with adversarial feedback)
            loss_adv = -discriminator(c_all, inputs["s"])
            loss = loss_net + loss_adv
            
            optimizer_net.zero_grad()
            loss.backward()
            optimizer_net.step()
            
            epoch_loss += loss.item()
            
        pbar.set_postfix({"Loss": f"{epoch_loss/n_batches_in_epoch:.4f}"})
            
    # 5. Extract Embeddings
    print("Extracting cell embeddings...")
    net.eval()
    all_embeddings = []
    
    with th.no_grad():
        for b_idx in range(int(np.ceil(n_cells / o.batch_size))):
            indices = np.arange(b_idx * o.batch_size, min((b_idx + 1) * o.batch_size, n_cells))
            inputs = {
                "raw": {
                    "rna": th.tensor(rna_data[indices]).cuda(),
                    "adt": th.tensor(adt_data[indices]).cuda(),
                },
                "s": {
                    "joint": th.tensor(batch_ids[indices]).unsqueeze(1).cuda(),
                    "rna": th.tensor(batch_ids[indices]).unsqueeze(1).cuda(),
                    "adt": th.tensor(batch_ids[indices]).unsqueeze(1).cuda(),
                },
                "e": {
                    "rna": th.ones(len(indices), n_genes).cuda(),
                    "adt": th.ones(len(indices), n_proteins).cuda(),
                }
            }
            # conditioned on all observed modalities, get z["raw"]
            _, _, _, _, z, _, _, *_ = net.sct(inputs)
            all_embeddings.append(z["raw"].cpu().numpy())
            
    joint_emb = np.concatenate(all_embeddings, axis=0)
    
    # 6. Save results
    os.makedirs(args.save_path, exist_ok=True)
    emb_file = os.path.join(args.save_path, "embedding.h5")
    with h5py.File(emb_file, 'w') as f:
        f.create_dataset('data', data=joint_emb)
        
    elapsed_time = time.time() - start_time
    print(f"Done in {elapsed_time:.1f}s. Saved embedding to {emb_file}")
    
    # Save metrics.json for logger
    metrics_file = os.path.join(args.save_path, "metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump({
            "time": elapsed_time,
            "Embedding_Dim": joint_emb.shape[1],
            "Actual_Epochs": args.epochs
        }, f)

if __name__ == "__main__":
    main()
