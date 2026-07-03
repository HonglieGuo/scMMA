import os
import sys
from os.path import join as pj
from glob import glob
sys.path.append(os.path.abspath('./MINERVA'))
from modules import utils
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--task', type=str, default='dm_sub10',
    help="experiment name")
o = parser.parse_args()

base_path = pj("./result/preprocess", o.task)
base_dirs = glob(pj(base_path, "subset_*"))


for base_dir in base_dirs:
    
    # Specify directories
    in_dir = pj(base_dir, "mat")
    out_dir = pj(base_dir, "vec")
    utils.mkdirs(out_dir, remove_old = True)
    print("\nDirectory: %s" % (in_dir))
    
    # Load and save data
    exp_names = glob(pj(in_dir, '*.csv'))  # get filenames
    for i, exp_name in enumerate(exp_names):
        # load
        exp = utils.load_csv(exp_name)
        mode = os.path.splitext(os.path.basename(exp_name))[0]
        cell_num = len(exp) - 1
        feat_num = len(exp[0]) - 1
        print("Spliting %s matrix: %d cells, %d features" % (mode, cell_num, feat_num))
        
        # save
        out_mode_dir = pj(out_dir, mode)
        utils.mkdirs(out_mode_dir, remove_old=True)
        cell_name_fmt = utils.get_name_fmt(cell_num) + ".csv"

        cell_name_fmt = pj(out_mode_dir, cell_name_fmt)
        for cell_id in range(cell_num):
            cell_name = cell_name_fmt % (cell_id + 1)
            utils.save_list_to_csv([exp[cell_id+1][1:]], cell_name)
        
