import os
import time
import h5py
import anndata
import argparse
import MultiMAP
import numpy as np
import scanpy as sc
from anndata import AnnData
from scipy.sparse import csr_matrix

begin_time = time.time()
parser = argparse.ArgumentParser("MultiMAP")
parser.add_argument('--path1', nargs='+', default=['NULL'], help='path to train gene')
parser.add_argument('--path2', nargs='+', default=['NULL'], help='path to train peak')
parser.add_argument('--path3', nargs='+', default=['NULL'], help='path to train gene activity score')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
args = parser.parse_args()

# The MultiMAP script for diagonal integration requires RNA and ATAC data as input, where ATAC needs to be transformed into gene activity score. The output is a joint embedding (dimensionality reduction).
# run commond for MultiMAP
# python main_MultiMAP.py --path1 "../../data/dataset_final/D27/rna.h5" --path2 "../../data/dataset_final/D27/atac_peak.h5" --path3 "../../data/dataset_final/D27/atac_gas.h5" --save_path "../../result/embedding/diagonal integration/D27/MultiMAP/"

def load_and_concat(paths):
    adatas = []
    for p in paths:
        if p == 'NULL': continue
        with h5py.File(p, "r") as f:
            X = np.mat(np.array(f['matrix/data']).transpose())
            adata = AnnData(X=X)
            adata.X = csr_matrix(np.matrix(adata.X))
            adatas.append(adata)
    if len(adatas) == 0:
        return None
    return anndata.concat(adatas, join='outer')

def runMultiMAP(path1_list, path2_list, path3_list=['NULL']):
    rna = load_and_concat(path1_list)
    mod2 = load_and_concat(path2_list)
    
    rna_pca = rna.copy()
    sc.pp.scale(rna_pca)
    sc.pp.pca(rna_pca)
    rna.obsm['X_pca'] = rna_pca.obsm['X_pca'].copy()

    path2_first = path2_list[0] if isinstance(path2_list, list) else path2_list
    
    if "adt" in path2_first.lower() or "protein" in path2_first.lower():
        mod2_pca = mod2.copy()
        sc.pp.scale(mod2_pca)
        sc.pp.pca(mod2_pca)
        mod2.obsm['X_pca'] = mod2_pca.obsm['X_pca'].copy()
        adata = MultiMAP.Integration([rna, mod2], ['X_pca', 'X_pca'])
    else:
        MultiMAP.TFIDF_LSI(mod2)
        atac_genes = load_and_concat(path3_list)
        if atac_genes is not None:
            atac_genes.obsm['X_lsi'] = mod2.obsm['X_lsi'].copy()
            adata = MultiMAP.Integration([rna, atac_genes], ['X_pca', 'X_lsi'])
        else:
            adata = MultiMAP.Integration([rna, mod2], ['X_pca', 'X_lsi'])

    result = adata.obsm['X_multimap']
    return result

result = runMultiMAP(args.path1, args.path2, args.path3)
end_time = time.time()
all_time = end_time - begin_time
print(all_time)

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print("create path")
else:
    print("the path exits")
file = h5py.File(args.save_path+"/embedding.h5", 'w')
file.create_dataset('data', data=result)
file.close()
np.savetxt(args.save_path+"/time.csv", [all_time], delimiter=",")
