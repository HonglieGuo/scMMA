import os
from os.path import join as pj
import argparse
import sys
sys.path.append(os.path.abspath('./MINERVA'))
from modules import utils
import numpy as np
import scib
import scib.metrics as me
import anndata as ad
import scipy
import pandas as pd
import re
from glob import glob
import copy

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='dm_sub10')
parser.add_argument('--experiment', type=str, default='e0')
parser.add_argument('--model', type=str, default='default')
parser.add_argument('--method', type=str, default='embed')
o, _ = parser.parse_known_args()  # for python interactive
test_dir = "predict"
start = 999
end = 1000
step = 100

# %%
print(os.getcwd())
cfg_task = re.sub("_vd.*|_vt.*|_atlas|_generalize|_transfer|_ref_.*", "", o.task)
data_config = utils.load_toml("./MINERVA/configs/data.toml")[cfg_task]
for k, v in data_config.items():
    vars(o)[k] = v
model_config = utils.load_toml("./MINERVA/configs/model.toml")["default"]
if o.model != "default":
    model_config.update(utils.load_toml("./MINERVA/configs/model.toml")[o.model])
for k, v in model_config.items():
    vars(o)[k] = v
o.s_joint, o.combs, *_ = utils.gen_all_batch_ids(o.s_joint, o.combs)

# %%
# Load cell type labels
labels = []
batch = []
labels_dir = []
for raw_data_dir in o.raw_data_dirs:
    for s in o.s_joint:
        labels_dir += glob(pj("./result/preprocess", o.task, "subset_" + str(s), "labels.csv"))

p = 0
for l in labels_dir:
    label = utils.load_csv(l)[1:]
    labels += utils.transpose_list(label)[6]
    batch += [str(p)] * len(label)
    p += 1

labels = np.array(labels)
batch = np.array(batch)

print(list(set(labels)))
print(list(set(batch)))


def bio_evaluate(o, pred, labels, result_dir):
    if o.method in ["embed", "stabmap", "scvaeit", "multigrate"]:
        output_type = "embed"
    else:
        output_type = "graph"
    else:
        assert False, o.method+": invalid method!"

    embed = "X_emb"
    batch_key = "batch"
    label_key = "label"
    cluster_key = "cluster"
    si_metric = "euclidean"
    subsample = 1.0
    verbose = False

    c = pred["z"]["joint"][:, :o.dim_c]
    s = pred["s"]["joint"]

    if o.method == "embed":
        adata = ad.AnnData(c)
        adata.obsm[embed] = c
        adata.obs[batch_key] = s.astype(str)
        adata.obs[batch_key] = adata.obs[batch_key].astype("category")
        adata.obs[label_key] = labels
        adata.obs[label_key] = adata.obs[label_key].astype("category")
    elif o.method in ["stabmap", "multigrate"]:
        adata = ad.AnnData(c*0)
        embeddings = utils.load_csv(pj(result_dir, "embeddings.csv"))
        adata.obsm[embed] = np.array(embeddings)[1:, 1:].astype(np.float32)
        adata.obs[batch_key] = s.astype(str)
        adata.obs[batch_key] = adata.obs[batch_key].astype("category")
        adata.obs[label_key] = labels
        adata.obs[label_key] = adata.obs[label_key].astype("category")
    elif o.method in ["scvaeit"]:
        adata = ad.AnnData(c*0)
        embeddings = utils.load_csv(pj(result_dir, "embeddings.csv"))
        adata.obsm[embed] = np.array(embeddings).astype(np.float32)
        adata.obs[batch_key] = s.astype(str)
        adata.obs[batch_key] = adata.obs[batch_key].astype("category")
        adata.obs[label_key] = labels
        adata.obs[label_key] = adata.obs[label_key].astype("category")
    else:
        adata = ad.AnnData(c*0)
        adata.obs[batch_key] = s.astype(str)
        adata.obs[batch_key] = adata.obs[batch_key].astype("category")
        adata.obs[label_key] = labels
        adata.obs[label_key] = adata.obs[label_key].astype("category")
        adata.obsp["connectivities"] = scipy.io.mmread(pj(result_dir, "connectivities.mtx")).tocsr()
        adata.uns["neighbors"] = {'connectivities_key': 'connectivities'}

    results = {}
    print(adata)

    print('clustering...')
    res_max, nmi_max, nmi_all = scib.clustering.opt_louvain(adata, label_key=label_key,
        cluster_key=cluster_key, function=me.nmi, use_rep=embed, verbose=verbose, inplace=True)
    
    results['NMI'] = me.nmi(adata, group1=cluster_key, group2=label_key, method='arithmetic')
    print("NMI: " + str(results['NMI']))

    results['ARI'] = me.ari(adata, group1=cluster_key, group2=label_key)
    print("ARI: " + str(results['ARI']))

    type_ = "knn" if output_type == "graph" else None
    results['kBET'] = me.kBET(adata, batch_key=batch_key, label_key=label_key, embed=embed, 
        type_=type_, verbose=verbose, return_df = False)
    print("kBET: " + str(results['kBET']))

    results['il_score_asw'] = me.isolated_labels(adata, label_key=label_key, batch_key=batch_key,
        embed=embed, cluster=False, verbose=verbose)
    print("il_score_asw: " + str(results['il_score_asw']))

    results['graph_conn'] = me.graph_connectivity(adata, label_key=label_key)
    print("graph_conn: " + str(results['graph_conn']))

    results['iLISI'] = me.ilisi_graph(adata, batch_key=batch_key, type_="knn",
        subsample=subsample*100, n_cores=1, verbose=verbose)
    print("iLISI: " + str(results['iLISI']))

    results = {k: float(v) for k, v in results.items()}
    results['batch_score'] = np.nanmean([results['iLISI'], results['graph_conn'], results['kBET']])
    results['bio_score'] = np.nanmean([results['NMI'], results['ARI'], results['il_score_asw'])

    df = pd.DataFrame({
        'iLISI':          [results['iLISI']],
        'graph_conn':     [results['graph_conn']],
        'kBET':           [results['kBET']],
        'batch_score':    [results['batch_score']],
        'NMI':            [results['NMI']],
        'ARI':            [results['ARI']],
        'il_score_asw':    [results['il_score_asw']],
        'bio_score':      [results['bio_score']],
    })
    print(df)

    return df


# %%
o.mods = ["rna", "adt"]
df_batch_bio_embed = []

for i in list(range(start, end, step)):
    init_model = "sp_0000{:04d}".format(i)
    result_dir = pj("result", "comparison", o.task, o.experiment)
    utils.mkdirs(result_dir, remove_old = False)
    # Load predicted latent variables
    o.pred_dir = pj("result", o.task, o.experiment, o.model, test_dir, init_model)
    pred = utils.load_predicted(o)
    df_bio = bio_evaluate(o, pred, labels, result_dir)
    df_bio["Model"] = "Step_" + "{:04d}".format(i + 1)
    df_batch_bio_embed.append(df_bio)

# DM
df_bio_cat = pd.concat(df_batch_bio_embed, axis = 0)
df_bio_mean_cat = copy.deepcopy(df_bio_cat)
df_bio_mean_cat["bio_score"] = df_bio_cat[["NMI", "ARI", "il_score_asw"]].mean(axis = 1)
df_bio_mean_cat = df_bio_mean_cat[["Model", "NMI", "ARI", "il_score_asw", "bio_score"]]
df_bio_mean_cat_sorted = df_bio_mean_cat.sort_values("bio_score", ascending = False, inplace = False)
df_bio_mean_cat.to_csv(pj(result_dir, "metrics_bio_train.csv"), index = False)
df_bio_mean_cat

# # SLN
# df_bio_cat = pd.concat(df_batch_bio_embed, axis = 0)
# df_bio_mean_cat = copy.deepcopy(df_bio_cat)
# df_bio_mean_cat["batch_score"] = df_bio_cat[["iLISI", "graph_conn", "kBET"]].mean(axis = 1)
# df_bio_mean_cat["bio_score"] = df_bio_cat[["NMI", "ARI", "il_score_asw", "cLISI"]].mean(axis = 1)
# df_bio_mean_cat = df_bio_mean_cat[["Model", "NMI", "ARI", "il_score_asw", "cLISI", "bio_score", "iLISI", "graph_conn", "kBET", "batch_score"]]
# df_bio_mean_cat.to_csv(pj(result_dir, "metrics_bio_train.csv"), index = False)
# print(df_bio_mean_cat)
