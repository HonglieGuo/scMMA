# %%
import os
from os.path import join as pj
import argparse
import sys
sys.path.append("modules")
import pandas as pd

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='dm_sub10')
parser.add_argument('--experiment', type=str, default='e0')
parser.add_argument('--model', type=str, default='default')
parser.add_argument('--method', type=str, default='embed')
o, _ = parser.parse_known_args()  # for python interactive
test_dir = "predict"

# %%
df_bio = pd.read_csv(pj("result", "comparison", o.task, o.experiment, "metrics_bio_train.csv"))
df_mod = pd.read_csv(pj("result", "comparison", o.task, o.experiment, "metrics_mod_train.csv"))
df_impu = pd.read_csv(pj("result", "comparison", o.task, o.experiment, "metrics_impu_train.csv"))

# %%
merged_df = pd.merge(df_bio, df_mod, on='Model')
merged_df = pd.merge(merged_df, df_impu, on='Model')

# %%
# SLN
merged_df["overall_score"] =  0.1 * merged_df["impu_score"] + \
                                0.1 * merged_df["mod_score"] + \
                                0.4 * merged_df["batch_score"] + \
                                0.4 * merged_df["bio_score"]
merged_df
