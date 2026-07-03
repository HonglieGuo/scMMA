import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import scanpy as sc
from sklearn.preprocessing import LabelEncoder
from scipy.stats import median_abs_deviation

def load_data(
    omic_dir_lst,  # rna, atac, adt
    omic_type,
    reserve_dims,
):
    """
    Load data from h5ad files.

    :param omic_dir_lst: datasets directory, including RNA, ATAC, ADT and such.
    :param reserve_dims: the dimensions of highly_variable_genes
    """
    
    omic_adatas = []
    dims = []
    xs = []
    dims = []
    print('++'*10)
    print(omic_dir_lst)
    
    # 's2d1', 's2d4', 's2d5', 's3d3', 's3d6', 's3d7', 's3d10', 's4d1', 's4d8', 's4d9', 
    # omic_adatas[i] = omic_adatas[i][omic_adatas[i].obs['batch'].isin(['s1d1', 's1d2', 's1d3',])]
    
    for i in range(len(omic_dir_lst)):
        omic_adatas.append(sc.read_h5ad(omic_dir_lst[i]))
        
    labels = omic_adatas[0].obs['cell_type'].values
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    labels = label_encoder.transform(labels)
    
    for i in range(len(omic_dir_lst)):
        # omic_adatas.append(sc.read_h5ad(omic_dir_lst[i]))
        
        # sc.pp.filter_genes(omic_adatas[i], min_cells=2,)
        # if omic_adatas[i].X.max() >= 16:
        #     sc.pp.normalize_total(omic_adatas[i], target_sum=1e4)
        #     sc.pp.log1p(omic_adatas[i])
        # sc.pp.highly_variable_genes(omic_adatas[i], n_top_genes=reserve_dims[i], subset=True)
        # omic_adatas[i] = omic_adatas[i][:, omic_adatas[i].var.highly_variable]
        
        omic_adatas[i] = fn1(omic_adatas[i], omic_type[i], reserve_dims[i])
        dims.append(omic_adatas[i].shape[1])
        
        try:
            x = omic_adatas[i].X.toarray().astype(np.float32)
        except:
            x = omic_adatas[i].X.astype(np.float32)
        xs.append(x)
    
    full_dataset = MultiOmicsFullDataset(
        xs,
        labels,
        get_paired_lqc_idx(omic_adatas, omic_type),
        get_unpaired_lqc_idx(omic_adatas),
    )
    
    return full_dataset, dims, len(dims), xs[0].shape[0], len(np.unique(labels))


class MultiOmicsFullDataset(Dataset):
    def __init__(
        self,
        xs,
        label,
        lqc_paired_idx,
        lqc_unpaired_idx,
    ):
        super().__init__()
        self.xs = xs
        self.label = label
        self.lqc_paired_idx = lqc_paired_idx
        self.lqc_unpaired_idx = lqc_unpaired_idx

    def full_data(self):
        return [torch.from_numpy(u) for u in self.xs], self.label
    
    def paired_clean_data(self):
        return [torch.from_numpy(u[~self.lqc_paired_idx]) for u in self.xs], self.label[~self.lqc_paired_idx]
    
    def unpaired_clean_data(self):
        return [torch.from_numpy(self.xs[i][~self.lqc_unpaired_idx[i]]) for i in range(len(self.xs))]
    
    def __len__(self):
        return self.xs[0].shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(u[idx]) for u in self.xs], self.label[idx]

class MultiOmicsCleanDataset(Dataset):
    def __init__(
        self,
        xs_cleaned,
        label_cleaned,
        # lqc_idx,
    ):
        super().__init__()
        self.xs_cleaned = xs_cleaned
        self.label_cleaned = label_cleaned
        # self.lqc_idx = lqc_idx

    def full_data(self):
        return [torch.from_numpy(u) for u in self.xs_cleaned], self.label_cleaned
    
    # def clean_data(self):
    #     return [torch.from_numpy(u[~self.lqc_idx]) for u in self.xs], self.label[~self.lqc_idx]
    
    def __len__(self):
        return self.xs_cleaned[0].shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(u[idx]) for u in self.xs_cleaned], self.label_cleaned[idx]
    

def get_paired_lqc_idx(omic_adatas, omic_type): # op='&'
    ### strictly filter outlier cells
    # select_lqc = np.array([0.0]*omic_adatas[0].shape[0])
    # for i in range(len(omic_adatas)):
    #     adata = omic_adatas[i]
    #     if omic_type[i] == 'rna':
    #         weight = 1.0
    #     elif omic_type[i] == 'atac':
    #         weight = 0.4
    #     elif omic_type[i] == 'adt':
    #         weight = 0.8
    #     select_lqc = select_lqc + (adata.obs['low_quality_cells'].values != 'outlier').astype(float) * weight 
    # select_lqc = select_lqc <= 1.1
    
    select_lqc = np.array([False]*omic_adatas[0].shape[0])
    for adata in omic_adatas:
        select_lqc = select_lqc | (adata.obs['low_quality_cells'].values == 'outlier')
        
    # if op == '&':
    #     select_lqc = np.array([True]*omic_adatas[0].shape[0])
    #     for adata in omic_adatas:
    #         select_lqc = select_lqc & (adata.obs['low_quality_cells'].values == 'outlier')
    # elif op == '|':
    #     select_lqc = np.array([False]*omic_adatas[0].shape[0])
    #     for adata in omic_adatas:
    #         select_lqc = select_lqc | (adata.obs['low_quality_cells'].values == 'outlier')
    return select_lqc


def get_unpaired_lqc_idx(omic_adatas):
    select_lqc_ls = []
    for adata in omic_adatas:
        select_lqc_ls.append(adata.obs['low_quality_cells'].values == 'outlier')
    return select_lqc_ls


def fn1(adata, omic_type, reserve_dim):
    adata_filtered = filter_low_quality_cells(adata.copy(), omic_type, reserve_dim+1000)
    # 找出被过滤的细胞
    filtered_cells = set(adata.obs.index) - set(adata_filtered.obs.index)

    # 将被过滤的细胞标记为 True
    adata.obs['low_quality_cells'] = 'normal'
    adata.obs.loc[list(filtered_cells), 'low_quality_cells'] = 'outlier'
    if omic_type != 'adt':
        adata = adata[:, adata_filtered.var.highly_variable.index]
    if adata.X.max() >= 16:
        sc.pp.normalize_total(adata, target_sum=1e4)    
        sc.pp.log1p(adata)
    
    if omic_type != 'adt':
        sc.pp.highly_variable_genes(adata, n_top_genes=reserve_dim, subset=True)
    
    del filtered_cells
    return adata


def filter_low_quality_cells(adata, omic_type, reserve_dim, n_pca_components=40):  
    """
    Low-Quality Cell Filtering Based on PCA and Distance Metrics

    Parameters:
    adata: An AnnData object containing preprocessed single-cell data.
    reserve_dim: The number of dimensions to be reserved after filtering.
    n_pca_components: The number of principal components to be used in PCA.
    
    Returns:
    adata: The filtered AnnData object, with a low_quality_cells field marking anomalous cells.
    """
    
    if omic_type != 'adt':
        sc.pp.filter_genes(adata, min_cells=2,)
        sc.pp.filter_cells(adata, min_genes=300)
        
        # 计算质控指标
        adata.obs['n_counts'] = np.ravel(adata.X.sum(axis=1))  # 总UMI计数
        try:
            adata.obs['n_genes'] = np.ravel(adata.X.getnnz(axis=1))  # 检测到的基因数
        except:
            adata.obs['n_genes'] = np.ravel(np.count_nonzero(adata.X, axis=1))
        adata.obs['zero_rate'] = 1 - adata.obs['n_genes'] / adata.n_vars  # 零值比例

        # MAD异常值检测
        def mad_filter(values, n_mads=3):
            median = np.median(values)
            mad = np.median(np.abs(values - median))
            return (values >= median - n_mads*mad) & (values <= median + n_mads*mad)
        
        valid = mad_filter(adata.obs['n_counts']) & mad_filter(adata.obs['n_genes'])
        adata = adata[valid]#.copy()
    
    if adata.X.max() >= 16:
        sc.pp.normalize_total(adata, target_sum=1e4)    
        sc.pp.log1p(adata)
    
    if omic_type != 'adt':
        sc.pp.highly_variable_genes(adata, n_top_genes=reserve_dim, subset=True)
    
    try:
        # 对数据进行PCA降维  
        sc.pp.pca(adata, n_comps=n_pca_components, use_highly_variable=True, svd_solver='arpack')  

        # 获取PCA降维后的数据  
        X_pca = adata.obsm['X_pca']  
        outliers = fn3(X_pca)
    except:
        pass
    
    
    # 将异常细胞记录到AnnData的obs  
    adata.obs['low_quality_cell'] = False  
    
    try:
        adata.obs.loc[adata.obs.index[outliers], 'low_quality_cell'] = True 
    except:
        pass
    
    adata = adata[~adata.obs['low_quality_cell']]
    return adata  

def fn3(X_pca, q1=25, q3=75):
    # 计算所有细胞对的欧氏距离  
    from scipy.spatial.distance import pdist, squareform
    distances = pdist(X_pca, metric='euclidean')  
    distance_matrix = squareform(distances)  

    # 计算每个细胞到最近邻的距离（排除自身）  
    np.fill_diagonal(distance_matrix, np.inf)  # 将对角线置为无穷大  
    min_distances = np.min(distance_matrix, axis=1)  

    # 计算四分位数和异常值阈值  
    Q1 = np.percentile(min_distances, q1)  
    Q3 = np.percentile(min_distances, q3)  
    IQR = Q3 - Q1  
    threshold = Q3 + 1.5 * IQR  

    # 识别异常细胞  
    outliers = np.where(min_distances > threshold)[0]  
    return outliers

def filter_noise_latent_cells(hidden):
    """
    return: high quility cell indices.
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=40)
    hidden_reduced = pca.fit_transform(hidden)
    
    return list(set(np.arange(hidden.shape[0])) - set(fn3(hidden_reduced, 25, 75)))
