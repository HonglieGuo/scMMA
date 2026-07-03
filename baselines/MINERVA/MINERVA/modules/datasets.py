from os import path
from os.path import join as pj
import csv
import math
import numpy as np
import torch as th
from torch.utils.data import Dataset
from torch.distributions.dirichlet import Dirichlet
import modules.utils as utils
import re
import random
import math
import copy
import warnings
warnings.filterwarnings("ignore")

import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
robjects.r('library(scuttle)')


class MultimodalDataset(Dataset):

    def __init__(self, o, subset = 0, split = "train", comb = None, train_ratio = None, pair_mix = None):
        super(MultimodalDataset, self).__init__()
        
        config = utils.gen_data_config(o.task, o.reference)
        for kw, arg in config.items():
            setattr(self, kw, arg)

        _, self.combs, self.s, _ = utils.gen_all_batch_ids(self.s_joint, self.combs)
        
        assert subset < len(self.combs) == len(self.s), "Inconsistent subset specifications!"
        self.subset = subset
        self.comb = self.combs[subset] if comb is None else comb
        if train_ratio is not None: self.train_ratio = train_ratio
        self.s_subset = self.s[subset]
        self.o = o
        self.split = split
        self.pair_mix = pair_mix
        # self.s_drop_rate = s_drop_rate if split == "train" else 0
        
        base_dir = pj(self.o.data_dir, "subset_"+str(subset))
        self.in_dirs = {}
        self.masks = {}
        for m in self.comb:
            self.in_dirs[m] = pj(base_dir, "vec", m)
            if m in ["rna", "adt"]:
                mask = utils.load_csv(pj(base_dir, "mask", m+".csv"))[1][1:]
                self.masks[m] = np.array(mask, dtype=np.float32)

        filenames_list = []
        
        for in_dir in self.in_dirs.values():
            filenames_list.append(utils.get_filenames(in_dir, "csv"))
        cell_nums = [len(filenames) for filenames in filenames_list]
        assert cell_nums[0] > 0 and len(set(cell_nums)) == 1, \
            "Inconsistent cell numbers!"
        filenames = filenames_list[0]

        train_num = int(round(len(filenames) * self.train_ratio))
        if self.split == "train":
            self.filenames = filenames[:train_num]
            if "fusion" in self.o.pretext:
                self.pair_filenames = self.pair_mix[self.subset][:train_num]
            else:
                self.pair_filenames = None

        else:
            self.filenames = filenames[train_num:]
        self.size = len(self.filenames)


    def __getitem__(self, index):
        items = {"raw": {}, "s": {}, "e": {}}
        if self.split == "train":
            items.update({t: {} for t in self.o.pretext if t in ["mask", "noise", "downsample", "fusion"]})
            if "fusion" in self.o.pretext:
                items["mix1"], items["mix2"] = {}, {}

        for m, v in self.s_subset.items():
            items["s"][m] = np.array([v], dtype=np.int64)
        
        for m in self.comb:
            file_path = pj(self.in_dirs[m], self.filenames[index])
            v = np.array(utils.load_csv(file_path)[0])
            items["raw"][m] = v.astype(np.float32)
            if m in self.masks.keys():
                items["e"][m] = self.masks[m]

            if self.split == "train":
                tr = transformation(items["raw"][m], index, m, self.pair_filenames, self.pair_mix, self.o.data_dir)
                
                for t in self.o.pretext:

                    if t == "mask":
                        items[t][m] = tr.random_mask(self.o.mask_ratio)

                    if t == "noise":
                        items[t][m] = tr.random_gaussian_noise()

                    if t == "downsample":
                        matrix = robjects.r.matrix(np.array(items["raw"][m]))
                        row_names = robjects.StrVector([str(i) for i in range(1, items["raw"][m].shape[0] + 1)])
                        col_names = robjects.StrVector(["1"])
                        setattr(matrix, 'dimnames', robjects.ListVector({'x': row_names, 'y': col_names}))
                        downsampled = robjects.r['downsampleMatrix'](matrix, prop = random.random() * 0.5)
                        downsampled_array = np.array(robjects.r.matrix(downsampled)).flatten().tolist()
                        items[t][m] = th.from_numpy(np.array(downsampled_array, dtype = np.float32))

                    if t == "fusion":
                        items["mix_param"], items["mix1"][m], items["mix2"][m] = tr.mixup(m)
     
        return items


    def __len__(self):
        return self.size


class transformation():

    def __init__(self, cell_profile, index, m, pair_filenames, pair_mix, data_dir):
        self.cell_profile = copy.deepcopy(cell_profile)
        self.gene_num = len(self.cell_profile)
        self.pair_filenames = pair_filenames
        self.pair_mix = pair_mix
        if self.pair_filenames is not None:
            self.cell_num = len(self.pair_filenames)
        self.index = index
        self.data_dir = data_dir
        self.m = m
    
    def build_mask(self, masked_percentage: float):

        mask = np.concatenate([np.ones(int(self.gene_num * masked_percentage), dtype = bool), 
                            np.zeros(self.gene_num - int(self.gene_num * masked_percentage), dtype = bool)])
        np.random.shuffle(mask)

        return mask
    
    def random_mask(self, mask_ratio):

        self.cell_profile1 = copy.deepcopy(self.cell_profile)
        # create the mask for mutation
        mask = self.build_mask(np.random.uniform(0, mask_ratio))
        # do the mutation with prob
        self.cell_profile1[mask] = 0
        
        return self.cell_profile1


    def random_gaussian_noise(self, noise_percentage: float = 1.0, sigma: float = 0.2):

        self.cell_profile2 = copy.deepcopy(self.cell_profile)
        noise_percentage = np.random.uniform(0, noise_percentage)
        mask = self.build_mask(noise_percentage)

        # create the noise
        noise = np.random.normal(0, sigma, int(self.gene_num * noise_percentage))

        # do the mutation
        self.cell_profile2[mask] += noise
        # self.cell_profile2 = th.where(self.cell_profile2 < 0., 0., self.cell_profile2)
        self.cell_profile2[self.cell_profile2 < 0.] = 0.
        # self.cell_profile2 = np.round(self.cell_profile2)

        return self.cell_profile2


    def mixup(self, m):
        pair_path = self.pair_filenames[self.index]

        # apply instance mixup
        mix_data1 = np.array(utils.load_csv(pj(self.data_dir, "subset_" + pair_path[0].split("_")[0], 
                                               "vec", m, pair_path[0].split("_")[1]))).astype(np.float32)
        mix_data2 = np.array(utils.load_csv(pj(self.data_dir, "subset_" + pair_path[1].split("_")[0], 
                                               "vec", m, pair_path[1].split("_")[1]))).astype(np.float32)

        # prob of mix_id [0, 0, 0, 0.97, 0.03]
        mix_id = np.zeros(len(self.pair_mix.keys()))
        mix1 = int(pair_path[0].split("_")[0])
        mix2 = int(pair_path[1].split("_")[0])
        mix_raw_param = [mix1, mix2, 0]

        mix_raw_param = np.array(mix_raw_param).astype(np.float32)

        return mix_raw_param, mix_data1[0], mix_data2[0]
    


class MultiDatasetSampler(th.utils.data.sampler.Sampler):

    def __init__(self, dataset, batch_size=1, shuffle=True, max_dataset_size=10000):
        self.dataset = dataset
        self.batch_size = batch_size
        if shuffle:
            self.Sampler = th.utils.data.sampler.RandomSampler
        else:
            self.Sampler = th.utils.data.sampler.SequentialSampler
        self.number_of_datasets = len(dataset.datasets)
        self.largest_dataset_size = max([cur_dataset.size for cur_dataset in dataset.datasets])
        self.largest_dataset_size = min(self.largest_dataset_size, max_dataset_size)

    def __len__(self):
        return self.batch_size * math.ceil(self.largest_dataset_size / self.batch_size) * len(self.dataset.datasets)

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = self.Sampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size * self.number_of_datasets
        samples_to_grab = self.batch_size
        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        epoch_samples = self.largest_dataset_size * self.number_of_datasets

        final_samples_list = []  # this is a list of indexes from the combined dataset
        for _ in range(0, epoch_samples, step):
            for i in range(self.number_of_datasets):
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []
                for _ in range(samples_to_grab):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        # got to the end of iterator - restart the iterator and continue to get samples
                        # until reaching "epoch_samples"
                        sampler_iterators[i] = samplers_list[i].__iter__()
                        cur_batch_sampler = sampler_iterators[i]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                final_samples_list.extend(cur_samples)

        return iter(final_samples_list)
