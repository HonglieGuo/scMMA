# %%
import os
from os.path import join as pj
import argparse
import sys
sys.path.append(os.path.abspath('./MINERVA'))
from modules import utils
import numpy as np
import torch as th
import scib
import scib.metrics as me
import anndata as ad
import pandas as pd
import copy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from scipy.stats import pearsonr

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
data_dir = pj("./result/preprocess/", o.task)
data_config = utils.load_toml("./MINERVA/configs/data.toml")[o.task]
for k, v in data_config.items():
    vars(o)[k] = v
model_config = utils.load_toml("./MINERVA/configs/model.toml")["default"]
if o.model != "default":
    model_config.update(utils.load_toml("./MINERVA/configs/model.toml")[o.model])
for k, v in model_config.items():
    vars(o)[k] = v
o.s_joint, o.combs, *_ = utils.gen_all_batch_ids(o.s_joint, o.combs)


def mod_evaluate(o, pred, data_dir, result_dir):
    output_type = "embed"
    embed = "X_emb"
    batch_key = "batch"
    label_key = "label"
    mod_key = "modality"
    cluster_key = "cluster"
    si_metric = "euclidean"
    subsample = 0.5
    verbose = False

    results1 = {
        "asw_mod": {},
        "foscttm": {},
        "f1": {},
    }

    results2 = {
        "pearson_rna": {},
        "pearson_adt": {},
        "RMSLE_rna": {},
        "RMSLE_adt": {},
    }

    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')

    for batch_id in pred.keys():

        print("Processing batch: ", batch_id)
        z = pred[batch_id]["z"]
        x = pred[batch_id]["x"]
        x_trans = pred[batch_id]["x_trans"]
        mask_dir = pj(data_dir, "subset_"+str(batch_id), "mask")

        c = {m: v[:, :o.dim_c] for m, v in z.items()}
        c_cat = np.concatenate((c["rna"], c["adt"]), axis=0)
        mods_cat = ["rna"]*len(c["rna"]) + ["adt"]*len(c["adt"])

        label = utils.load_csv(pj("./result/preprocess/", o.task, "subset_" + str(batch_id), "labels.csv"))[1:]
        label = np.array(utils.transpose_list(label)[6])
        label_cat = np.tile(label, 2)

        assert len(c_cat) == len(mods_cat) == len(label_cat), "Inconsistent lengths!"

        batch = str(batch_id) # toml dict key must be str
        # print("Computing asw_mod")
        adata = ad.AnnData(c_cat)
        adata.obsm[embed] = c_cat
        adata.obs[mod_key] = mods_cat
        adata.obs[mod_key] = adata.obs[mod_key].astype("category")
        adata.obs[label_key] = label_cat
        adata.obs[label_key] = adata.obs[label_key].astype("category")

        results1["asw_mod"][batch] = me.silhouette_batch(adata, batch_key=mod_key,
            group_key=label_key, embed=embed, metric=si_metric, verbose=verbose)

        results1["foscttm"][batch] = {}
        results1["f1"][batch] = {}

        results2["pearson_rna"][batch] = {}
        results2["pearson_adt"][batch] = {}
        results2["RMSLE_rna"][batch] = {}
        results2["RMSLE_adt"][batch] = {}

        for m in c.keys() - {"joint"}:
            for m_ in set(c.keys()) - {m, "joint"}:
                k = m+"_to_"+m_
                # print(k+":")
                # print("Computing foscttm")
                results1["foscttm"][batch][k] = 1 - utils.calc_foscttm(th.from_numpy(c[m]), th.from_numpy(c[m_]))

                # print("Computing f1")
                knn.fit(c[m], label)
                label_pred = knn.predict(c[m_])
                # cm = confusion_matrix(label, label_pred, labels=knn.classes_)
                results1["f1"][batch][k] = f1_score(label, label_pred, average='micro')

                if m_ in ["rna", "adt"]:
                    mask = np.array(utils.load_csv(pj(mask_dir, m_+".csv"))[1][1:]).astype(bool)
                    x_gt = x[m_][:, mask].reshape(-1)
                    x_pred = x_trans[k][:, mask].reshape(-1)
                    # print("Computing pearson_"+m_)
                    results2["pearson_"+m_][batch][k] = pearsonr(x_gt, x_pred)[0]
                    # results2["pearson_"+m_][batch] = [pearsonr(x_gt[i], x_pred[i])[0] for i in np.arange(x_pred.shape[0])]
                    # results2["RMSLE_"+m][batch] = [np.sqrt(np.mean(np.power(np.log(x_gt[i] + 1) - np.log(x_pred[i] + 1), 2))) for i in np.arange(x_pred.shape[0])]
                    results2["RMSLE_"+m_][batch][k] = np.sqrt(np.mean(np.power(np.log(x_gt + 1) - np.log(x_pred + 1), 2)))


    results1_avg = {metric: np.mean(utils.extract_values(v)) for metric, v in results1.items()}
    df1 = pd.DataFrame({
        'ASW_mod':          [results1_avg['asw_mod']],
        'FOSCTTM':          [results1_avg['foscttm']],
        'Label_transfer':   [results1_avg['f1']]
    })
    print(df1)

    results2_avg = {metric: np.mean(utils.extract_values(v)) for metric, v in results2.items()}
    df2 = pd.DataFrame({
        'Pearson_RNA':      [results2_avg['pearson_rna']],
        'Pearson_ADT':      [results2_avg['pearson_adt']],
        'RMSLE_RNA':        [results2_avg['RMSLE_rna']],
        'RMSLE_ADt':        [results2_avg['RMSLE_adt']]
    })
    print(df2)

    return df1, df2



# %%
o.mods = ["rna", "adt"]
df_mod_embed, df_impu_x = [], []
for i in list(range(start, end, step)):
    init_model = "sp_0000{:04d}".format(i)
    result_dir = pj("result", "comparison", o.task, o.experiment)
    # Load predicted latent variables
    o.pred_dir = pj("result", o.task, o.experiment, o.model, "predict", init_model)
    pred = utils.load_predicted(o, mod_latent=True, translate=True, input=True,impute=True, group_by="subset")
    df_mod, df_impu = mod_evaluate(o, pred, data_dir, result_dir)
    df_mod["Model"] = "Step_" + "{:04d}".format(i + 1)
    df_impu["Model"] = "Step_" + "{:04d}".format(i + 1)
    df_mod_embed.append(df_mod)
    df_impu_x.append(df_impu)


# %%
df_mod_cat = pd.concat(df_mod_embed, axis = 0)
df_mod_mean_cat = copy.deepcopy(df_mod_cat)
df_mod_mean_cat["mod_score"] = df_mod_cat[["ASW_mod", "FOSCTTM", "Label_transfer"]].mean(axis = 1)
df_mod_mean_cat = df_mod_mean_cat[["Model", "ASW_mod", "FOSCTTM", "Label_transfer", "mod_score"]]
df_mod_mean_cat.to_csv(pj(result_dir, "metrics_mod_train.csv"), index = False)
print(df_mod_mean_cat)


# %%
df_impu_cat = pd.concat(df_impu_x, axis = 0)
df_impu_mean_cat = copy.deepcopy(df_impu_cat)
df_impu_mean_cat["impu_score"] = df_impu_cat[["Pearson_RNA", "Pearson_ADT"]].mean(axis = 1)
df_impu_mean_cat = df_impu_mean_cat[["Model", "Pearson_RNA", "Pearson_ADT", "impu_score"]]
df_impu_mean_cat.to_csv(pj(result_dir, "metrics_impu_train.csv"), index = False)
print(df_impu_mean_cat)
