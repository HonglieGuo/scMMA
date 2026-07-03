
import os
from os import path
from os.path import join as pj
import time
import random
import argparse

from tqdm import tqdm
import numpy as np
import torch as th
from torch import nn, autograd
import matplotlib.pyplot as plt
import umap
import re
import copy
import itertools

from modules import models, utils
from modules.datasets import MultimodalDataset
from modules.datasets import MultiDatasetSampler


parser = argparse.ArgumentParser()
## Base
parser.add_argument('--task', type=str, default='atlas',
    help="Choose a task")
parser.add_argument('--reference', type=str, default='',
    help="Choose a reference task")
parser.add_argument('--experiment', type=str, default='e0',
    help="Choose an experiment")
parser.add_argument('--rf_experiment', type=str, default='',
    help="Choose a reference experiment")
parser.add_argument('--model', type=str, default='default',
    help="Choose a model configuration")
parser.add_argument('--actions', type=str, nargs='+', default=['train'],
    help="Choose actions to run")
parser.add_argument('--method', type=str, default='midas',
    help="Choose an method to benchmark")
parser.add_argument('--init_model', type=str, default='',
    help="Load a trained model")
parser.add_argument('--init_from_ref', type=int, default=0,
    help="Load a model trained on the reference task")
parser.add_argument('--sample_num', type=int, default=0,
    help='Number of samples to be generated')
parser.add_argument('--input_mods', type=str, nargs='+', default=[],
    help="Input modalities for transformation")
## Training
parser.add_argument('--train_ratio', type=float, default=0.8,
    help='Proportion of data used for training to the total data')
parser.add_argument('--epoch_num', type=int, default=1000, 
    help='Number of epochs to train')
parser.add_argument('--batch_size', type=int, default=128,
    help='Number of samples in a mini-batch')
parser.add_argument('--lr', type=float, default=1e-4,
    help='Learning rate')
parser.add_argument('--grad_clip', type=float, default=-1,
    help='Gradient clipping value')
parser.add_argument('--s_drop_rate', type=float, default=0.1,
    help="Probility of dropping out subject ID during training")
parser.add_argument('--seed', type=int, default=1234,
    help="Set the random seed to reproduce the results")
parser.add_argument('--use_shm', type=int, default=1,
    help="Use shared memory to accelerate training")
## Debugging
parser.add_argument('--print_iters', type=int, default=-1,
    help="Iterations to print training messages")
parser.add_argument('--log_epochs', type=int, default=100,
    help='Epochs to log the training states')
parser.add_argument('--save_epochs', type=int, default=100,
    help='Epochs to save the latest training states (overwrite previous ones)')
parser.add_argument('--time', type=int, default=0, choices=[0, 1],
    help='Time the forward and backward passes')
parser.add_argument('--debug', type=int, default=0, choices=[0, 1],
    help='Print intermediate variables')
## Tasks
parser.add_argument('--pretext', type=str, nargs = '+', choices=['mask', 'noise', 'downsample', 'fusion'], default=['mask', 'noise', 'downsample', 'fusion'],
    help="Pretext in training model")
parser.add_argument('--mask_ratio', type=float, default='0.3',
    help="Mask ratio of features in each cells")
# o, _ = parser.parse_known_args()  # for python interactive
o = parser.parse_args()
o.pretext.insert(0, 'raw')
print(o.pretext)

# Initialize global varibles
data_config = None
net = None
discriminator = None
optimizer_net = None
optimizer_disc = None
benchmark = {
    "train_loss": [],
    "test_loss": [],
    "foscttm": [],
    "epoch_id_start": 0
}


def main():
    initialize()
    if o.actions == "print_model":
        print_model()
    if "train" in o.actions:
        train()
    if "test" in o.actions:
        test()
    if "save_input" in o.actions:
        predict(joint_latent=False, input=True)
    if "predict_all" in o.actions:
        predict(mod_latent=True, impute=True, batch_correct=True, translate=True, input=True)
    if "predict_joint" in o.actions:
        predict()
    if "predict_all_latent" in o.actions:
        predict(mod_latent=True)
    if "impute" in o.actions:
        predict(impute=True, input=True)
    if "impute2" in o.actions:
        predict(impute=True)
    if "translate" in o.actions:
        predict(translate=True, input=True)
    if "batch_correct" in o.actions:
        predict(batch_correct=True, input=True)
    if "predict_all_latent_bc" in o.actions:
        predict(mod_latent=True, batch_correct=True, input=True)
        

def initialize():
    init_seed()
    init_dirs()
    load_data_config()
    load_model_config()
    get_gpu_config()
    init_model()


def init_seed():
    if o.seed >= 0:
        np.random.seed(o.seed)
        th.manual_seed(o.seed)
        th.cuda.manual_seed_all(o.seed)


def init_dirs():
    if o.use_shm == 1:
        o.data_dir = pj("./result/preprocess", o.task)
    elif o.use_shm == 2:
        o.data_dir = pj("./result/preprocess", o.task, "train")
    elif o.use_shm == 3:
        o.data_dir = pj("./result/preprocess", o.task, "test")
    else:
        o.data_dir = pj("./result/preprocess", o.task)

    o.result_dir = pj("result", o.task, o.experiment, o.model)
    o.pred_dir = pj(o.result_dir, "predict", o.init_model)
    o.train_dir = pj(o.result_dir, "train")
    o.debug_dir = pj(o.result_dir, "debug")
    utils.mkdirs([o.train_dir, o.debug_dir])
    print("Task: %s\nExperiment: %s\nModel: %s\n" % (o.task, o.experiment, o.model))


def load_data_config():
    
    if o.reference == '':
        o.dims_x, o.mods = get_dims_x(ref=0)
        o.ref_mods = o.mods
    else:
        _, o.mods = get_dims_x(ref=0)
        o.dims_x, o.ref_mods = get_dims_x(ref=1)
    o.mod_num = len(o.mods)

    if o.rf_experiment == '':
        o.rf_experiment = o.experiment
    
    global data_config

    data_config = utils.gen_data_config(o.task, o.reference)
    for k, v in data_config.items():
        vars(o)[k] = v
    if o.batch_size > 0:
        o.N = o.batch_size

    o.s_joint, o.combs, o.s, o.dims_s = utils.gen_all_batch_ids(o.s_joint, o.combs)

    o.dim_s = o.dims_s["joint"]
    o.dim_b = 2


def load_model_config():
    model_config = utils.load_toml("./MINERVA/configs/model.toml")["default"]
    if o.model != "default":
        model_config.update(utils.load_toml("./MINERVA/configs/model.toml")[o.model])
    for k, v in model_config.items():
        vars(o)[k] = v
    o.dim_z = o.dim_c + o.dim_b
    o.dims_dec_x = o.dims_enc_x[::-1]
    o.dims_dec_s = o.dims_enc_s[::-1]
    o.dims_h = {}
    for m, dim in o.dims_x.items():
        o.dims_h[m] = dim


def get_gpu_config():
    o.G = 1  # th.cuda.device_count()  # get GPU number
    assert o.N % o.G == 0, "Please ensure the mini-batch size can be divided " \
        "by the GPU number"
    o.n = o.N // o.G
    print("Total mini-batch size: %d, GPU number: %d, GPU mini-batch size: %d" % (o.N, o.G, o.n))


def init_model():
    """
    Initialize the model, optimizer, and benchmark
    """
    global net, discriminator, optimizer_net, optimizer_disc
    
    # Initialize models
    net = models.Net(o).cuda()
    discriminator = models.Discriminator(o).cuda()
    net_param_num = sum([param.data.numel() for param in net.parameters()])
    disc_param_num = sum([param.data.numel() for param in discriminator.parameters()])
    print('Parameter number: %.3f M' % ((net_param_num+disc_param_num) / 1e6))
    
    # Load benchmark
    if o.init_model != '':
        if o.init_from_ref == 0:
            fpath = pj(o.train_dir, o.init_model)
            savepoint_toml = utils.load_toml(fpath+".toml")
            benchmark.update(savepoint_toml['benchmark'])
            o.ref_epoch_num = savepoint_toml["o"]["ref_epoch_num"]
        else:
            fpath = pj("result", o.reference, o.rf_experiment, o.model, "train", o.init_model)
            print(fpath)
            benchmark.update(utils.load_toml(fpath+".toml")['benchmark'])
            o.ref_epoch_num = benchmark["epoch_id_start"]
    else:
        o.ref_epoch_num = 0

    # Initialize optimizers
    optimizer_net = th.optim.AdamW(net.parameters(), lr=o.lr)
    optimizer_disc = th.optim.AdamW(discriminator.parameters(), lr=o.lr)
    
    # Load models and optimizers
    if o.init_model != '':
        savepoint = th.load(fpath+".pt")

        if o.init_from_ref == 0:
            net.load_state_dict(savepoint['net_states'])
            discriminator.load_state_dict(savepoint['disc_states'])
            optimizer_net.load_state_dict(savepoint['optim_net_states'])
            optimizer_disc.load_state_dict(savepoint['optim_disc_states'])
        
        else:
            exclude_modules = ["s_enc", "s_dec"]
            pretrained_dict = {}
            for k, v in savepoint['net_states'].items():
                exclude = False
                for exclude_module in exclude_modules:
                    if exclude_module in k:
                        exclude = True
                        break
                if not exclude:
                    pretrained_dict[k] = v
            net_dict = net.state_dict()
            net_dict.update(pretrained_dict)
            net.load_state_dict(net_dict)
        print('Model is initialized from ' + fpath + ".pt")


def print_model():
    global net, discriminator
    with open(pj(o.result_dir, "model_architecture.txt"), 'w') as f:
        print(net, file=f)
        print(discriminator, file=f)


def get_dims_x(ref):
    if ref == 0:
        feat_dims = utils.load_csv(pj(o.data_dir, "feat", "feat_dims.csv"))
    else:
        if o.use_shm != 0:
            feat_dims = utils.load_csv(pj("./result/preprocess", o.reference, "test", "feat", "feat_dims.csv"))
        
    feat_dims = utils.transpose_list(feat_dims)
    
    dims_x = {}
    for i in range(1, len(feat_dims)):
        m = feat_dims[i][0]
        dims_x[m] = int(feat_dims[i][1])
    print("Input feature numbers: ", dims_x)

    mods = list(dims_x.keys())
    
    return dims_x, mods
    

def train():
    train_data_loader_cat = get_dataloader_cat("train", train_ratio = o.train_ratio)
    print("Length of train_loaders: " + str(len(train_data_loader_cat)) + '\n')
    for epoch_id in range(benchmark['epoch_id_start'], o.epoch_num):
        run_epoch(train_data_loader_cat, "train", epoch_id)
        check_to_save(epoch_id)


def get_dataloaders(split, train_ratio=None):
    data_loaders = {}
    for subset in range(len(o.s)):
        data_loaders[subset] = get_dataloader(subset, split, train_ratio=train_ratio)
    return data_loaders


def get_dataloader(subset, split, train_ratio = None):
    dataset = MultimodalDataset(o, subset, split, train_ratio=train_ratio, pair_mix=None)
    shuffle = True if split == "train" else False
    data_loader = th.utils.data.DataLoader(dataset, batch_size=o.N, shuffle=shuffle,
                                           num_workers=64, pin_memory=True)
    print("Subset: %d, modalities %s: %s size: %d" %
          (subset, str(o.combs[subset]), split, dataset.size))
    return data_loader


def get_dataloader_cat(split, train_ratio = None):
    datasets = []
    global mode
    base_indirs = {}
    mode = {}

    for b in range(len(o.s)):
        base_indirs[b] = pj(o.data_dir, "subset_" + str(b), "vec")
        # statistic modes of each subset
        mode[b] = [name for name in os.listdir(base_indirs[b])]

    if split == "train" and "fusion" in o.pretext:
        pair_mix = utils.fusion_data(o, mode, base_indirs, train_ratio)
    else:
        pair_mix = None
    for subset in range(len(o.s)):
        datasets.append(MultimodalDataset(o, subset, split, train_ratio=train_ratio, pair_mix = pair_mix))
        print("Subset: %d, modalities %s: %s size: %d" %  (subset, str(o.combs[subset]), split,
            datasets[subset].size))
    dataset_cat = th.utils.data.dataset.ConcatDataset(datasets)
    shuffle = True if split == "train" else False
    sampler = MultiDatasetSampler(dataset_cat, batch_size=o.N, shuffle=shuffle)
    data_loader = th.utils.data.DataLoader(dataset_cat, batch_size=o.N, sampler=sampler, 
        num_workers=64, pin_memory=True)
    return data_loader


def test():
    data_loaders = get_dataloaders()
    run_epoch(data_loaders, "test")


def run_epoch(data_loader, split, epoch_id=0):
    start_time = time.time()
    if split == "train":
        net.train()
        discriminator.train()
    elif split == "test":
        net.eval()
        discriminator.eval()
    else:
        assert False, "Invalid split: %s" % split

    losses = []
    raw_losses = []

    for i, data in enumerate(data_loader):

        total_loss, sum_losses, raw_loss, loss_adv = run_iter(split, epoch_id, i, data)
        losses.append(total_loss)
        raw_losses.append(raw_loss)
        if o.print_iters > 0 and (i+1) % o.print_iters == 0:
            print('%s\tepoch: %d/%d\tBatch: %d/%d\t%s_loss: %.2f'.expandtabs(3) % 
                  (o.task, epoch_id+1, o.epoch_num, i+1, len(data_loader), split, total_loss))

    loss_avg = np.nanmean(losses)
    raw_loss_avg = np.nanmean(raw_losses)
    
    if (epoch_id+1) % 10 == 0:
        print('%s\t%s\tepoch: %d/%d\t%s_loss: %.2f\t[raw_loss: %.2f, disc: %.2f]'.expandtabs(3) % 
              (o.task, o.experiment, epoch_id+1, o.epoch_num, split, loss_avg, raw_loss_avg, -loss_adv))
        for k in ['raw', 'mask', 'noise', 'downsample', 'fusion']:
            if k in sum_losses["loss_recon"].keys():
                if k != "fusion":
                    print('\t\t=> %s last minibatch - recon:%.3f, kld:%.3f'.expandtabs(3) %
                        (k, sum_losses["loss_recon"][k].item(),sum_losses["loss_kld_z"][k].item()))
                else:
                    print('\t\t=> %s last minibatch - recon:%.3f, kld:%.3f, SEM:%.3f'.expandtabs(3) %
                            (k, sum_losses["loss_recon"][k].item(), sum_losses["loss_kld_z"][k].item(), sum_losses["loss_SEM"][k].item()))
    benchmark[split+'_loss'].append((float(epoch_id), float(loss_avg)))
    return loss_avg


def run_iter(split, epoch_id, iter_id, inputs):

    if split == "train":
        skip = False

        if skip:
            return np.nan
        else:
            
            if "fusion" in o.pretext:
                # fusion
                alpha = th.tensor([1.0, 1.0])
                w = th.distributions.Dirichlet(alpha).sample((len(inputs["mix_param"]),))
                inputs["mix_param"][:, 2] = w[:, 0]
                for m in inputs["raw"].keys():
                    inputs["fusion"][m] = utils.fusion(inputs["mix_param"],inputs["mix1"][m], inputs["mix2"][m])

            # print(inputs)
            inputs = utils.convert_tensors_to_cuda(inputs)
            
            with autograd.set_detect_anomaly(o.debug == 1):
                sum_losses, loss_net, raw_loss, c_all = forward_net(inputs)
                # print(sum_losses)
                discriminator.epoch = epoch_id - o.ref_epoch_num
                K = 3

                for _ in range(K):
                    loss_disc = forward_disc(utils.detach_tensors(c_all), inputs["s"])
                    update_disc(loss_disc)
                # c = models.CheckBP('c')(c)
                loss_adv = forward_disc(c_all, inputs["s"])
                loss_adv = -loss_adv
                loss = loss_net + loss_adv
                update_net(loss)

    else:
        with th.no_grad():
            inputs = utils.convert_tensors_to_cuda(inputs)
            sum_losses, loss_net, raw_loss, c_all = forward_net(inputs)
            loss_adv = forward_disc(c_all, inputs["s"])
            loss_adv = -loss_adv
            loss = loss_net + loss_adv

    return loss.item(), sum_losses, raw_loss.item(), loss_adv


def forward_net(inputs):
    return net(inputs)


def forward_disc(c, s):
    return discriminator(c, s)


def update_net(loss):
    update(loss, net, optimizer_net)


def update_disc(loss):
    update(loss, discriminator, optimizer_disc)
    

def update(loss, model, optimizer):
    optimizer.zero_grad()
    loss.backward()
    if o.grad_clip > 0:
        nn.utils.clip_grad_norm_(model.parameters(), o.grad_clip)
    optimizer.step()


def check_to_save(epoch_id):
    if (epoch_id+1) % o.log_epochs == 0 or epoch_id+1 == o.epoch_num:
        save_training_states(epoch_id, "sp_%08d" % epoch_id)
    if (epoch_id+1) % o.save_epochs == 0 or epoch_id+1 == o.epoch_num:
        save_training_states(epoch_id, "sp_latest")


def save_training_states(epoch_id, filename):
    benchmark['epoch_id_start'] = epoch_id + 1
    utils.save_toml({"o": vars(o), "benchmark": benchmark}, pj(o.train_dir, filename+".toml"))
    th.save({"net_states": net.state_dict(),
             "disc_states": discriminator.state_dict(),
             "optim_net_states": optimizer_net.state_dict(),
             "optim_disc_states": optimizer_disc.state_dict()
            }, pj(o.train_dir, filename+".pt"))


def predict(joint_latent=True, mod_latent=False, impute=False, batch_correct=False, translate=False, 
            input=False):
    if translate:
        mod_latent = True
    print("Predicting ...")
    dirs = utils.get_pred_dirs(o, joint_latent, mod_latent, impute, batch_correct, translate, input)
    parent_dirs = list(set(map(path.dirname, utils.extract_values(dirs))))
    utils.mkdirs(parent_dirs, remove_old=True)
    utils.mkdirs(dirs, remove_old=True)
    data_loaders = get_dataloaders("test", train_ratio=0)
    net.eval()
    with th.no_grad():
        for subset_id, data_loader in data_loaders.items():
            print("Processing subset %d: %s" % (subset_id, str(o.combs[subset_id])))
            fname_fmt = utils.get_name_fmt(len(data_loader))+".csv"
            
            for i, data_in in enumerate(tqdm(data_loader)):
                data = utils.convert_tensors_to_cuda(data_in)
                
                # conditioned on all observed modalities
                if joint_latent:
                    x_r_pre, _, _, _, z, _, _, *_ = net.sct(data)  # N * K
                    utils.save_tensor_to_csv(z["raw"], pj(dirs[subset_id]["z"]["joint"], fname_fmt) % i)
                if impute:
                    x_r = models.gen_real_data(x_r_pre["raw"], sampling = True)
                    for m in o.mods:
                        utils.save_tensor_to_csv(x_r[m], pj(dirs[subset_id]["x_impt"][m], fname_fmt) % i)
                if input:  # save the input
                    for m in o.combs[subset_id]:
                        utils.save_tensor_to_csv(data["raw"][m].int(), pj(dirs[subset_id]["x"][m], fname_fmt) % i)

                # conditioned on each individual modalities
                if mod_latent:
                    for m in data["raw"].keys():
                        input_data = {
                            "raw": {m: data["raw"][m]},
                            "s": data["s"], 
                            "e": {}
                        }
                        if m in data["e"].keys():
                            input_data["e"][m] = data["e"][m]
                        x_r_pre, _, _, _, z, c, b, *_ = net.sct(input_data)  # N * K
                        utils.save_tensor_to_csv(z["raw"], pj(dirs[subset_id]["z"][m], fname_fmt) % i)
                        if translate: # single to double
                            x_r = models.gen_real_data(x_r_pre["raw"], sampling = True)
                            for m_ in set(o.mods) - {m}:
                                utils.save_tensor_to_csv(x_r[m_], pj(dirs[subset_id]["x_trans"][m+"_to_"+m_], fname_fmt) % i)

        if batch_correct:
            print("Calculating b_centroid ...")
            
            pred = utils.load_predicted(o)
            b = th.from_numpy(pred["z"]["joint"][:, o.dim_c:])
            s = th.from_numpy(pred["s"]["joint"])

            b_mean = b.mean(dim=0, keepdim=True)
            b_subset_mean_list = []
            for subset_id in s.unique():
                b_subset = b[s == subset_id, :]
                b_subset_mean_list.append(b_subset.mean(dim=0))
            b_subset_mean_stack = th.stack(b_subset_mean_list, dim=0)
            dist = ((b_subset_mean_stack - b_mean) ** 2).sum(dim=1)
            net.sct.b_centroid = b_subset_mean_list[dist.argmin()]
            net.sct.batch_correction = True
            
            print("Batch correction ...")
            for subset_id, data_loader in data_loaders.items():
                print("Processing subset %d: %s" % (subset_id, str(o.combs[subset_id])))
                fname_fmt = utils.get_name_fmt(len(data_loader))+".csv"
                
                for i, data in enumerate(tqdm(data_loader)):
                    data = utils.convert_tensors_to_cuda(data)
                    x_r_pre, *_ = net.sct(data)
                    x_r = models.gen_real_data(x_r_pre["raw"], sampling=True)
                    for m in o.mods:
                        utils.save_tensor_to_csv(x_r[m], pj(dirs[subset_id]["x_bc"][m], fname_fmt) % i)

main()
