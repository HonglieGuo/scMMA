from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import numpy as np
import torch
import scanpy as sc
import scib

def transform_label(labels):
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    labels = label_encoder.transform(labels)
    return labels


def cluster_acc(y_true, y_pred):
    assert y_pred.size == y_true.size
    # y_true = transform_label(y_true)
    # y_pred = transform_label(y_pred)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def evaluate_cmp(label, pred, z):
    label = transform_label(label)
    pred = transform_label(pred)
    
    # from sklearn.metrics import confusion_matrix
    # cm = confusion_matrix(label, pred)
    # # 找到每列的最大值行作为映射
    # max_indices = np.argmax(cm, axis=0)
    # pred = np.array([max_indices[i] for i in pred])

    acc = cluster_acc(label, pred)
    nmi = normalized_mutual_info_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    pur = purity(label, pred)
    hidden = sc.AnnData(X=z)
    hidden.obs['label'] = label
    hidden.obs['label'] = hidden.obs['label'].astype('category')
    hidden.obsm['X_pca'] = hidden.X.copy()
    casw = scib.metrics.silhouette(hidden, label_key='label', embed='X_pca')
    clisi = scib.metrics.clisi_graph(hidden, type_="embed", label_key='label', use_rep='X_pca')
    return nmi, ari, acc, pur, casw, clisi


def evaluate(label, pred, z, return_full_metric=False):
    label = transform_label(label)
    pred = transform_label(pred)
    
    # from sklearn.metrics import confusion_matrix
    # cm = confusion_matrix(label, pred)
    # max_indices = np.argmax(cm, axis=0)
    # pred = np.array([max_indices[i] for i in pred])

    acc = cluster_acc(label, pred)
    if return_full_metric == False:
        return acc
    nmi = normalized_mutual_info_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    pur = purity(label, pred)
    hidden = sc.AnnData(X=z)
    hidden.obs['label'] = label
    hidden.obs['label'] = hidden.obs['label'].astype('category')
    sc.tl.pca(hidden, svd_solver='arpack')
    pca_casw = scib.metrics.silhouette(hidden, label_key='label', embed='X_pca')
    pca_clisi = scib.metrics.clisi_graph(hidden, type_="embed", label_key='label', use_rep='X_pca')
    lda = LinearDiscriminantAnalysis()
    X_transformed = lda.fit_transform(hidden.obsm['X_pca'], pred)
    hidden.obsm['lda'] = X_transformed
    lda_casw = scib.metrics.silhouette(hidden, label_key='label', embed='lda')
    lda_clisi = scib.metrics.clisi_graph(hidden, type_="embed", label_key='label', use_rep='lda')
    return nmi, ari, acc, pur, pca_casw, pca_clisi, lda_casw, lda_clisi


def inference(loader, model, device, view, data_size, return_latent=False):
    model.eval()
    soft_vector = []
    target_vector = []
    labels_vector = []
    glb_vector = []
    model.eval()
    for step, (xs, y) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            # X_hat, Cs, P, Qs, _ = model.forward(xs)
            xrs, P, Qs, _, logits, glb = model.forward(xs)
            preds = torch.argmax(P, dim=1)
            q = sum(Qs) / view

        q = q.detach()
        soft_vector.extend(q.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
        target_vector.extend(preds.cpu().detach().numpy())
        if return_latent:
            glb_vector.append(glb.cpu().detach().numpy())

    if return_latent:
        glb_vector = np.concatenate(glb_vector, axis=0)
    labels_vector = np.array(labels_vector).reshape(data_size)
    target_pred = np.array(target_vector).reshape(data_size)
    soft_pred = np.argmax(np.array(soft_vector), axis=1)
    if not return_latent:
        return labels_vector, target_pred, soft_pred
    return labels_vector, target_pred, soft_pred, glb_vector #, Qs_vector


def valid(model, device, dataset, view, data_size, isprint=True, return_latent=False, return_full_metric=False):
    test_loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
    )
    
    labels_vector, target_pred, soft_pred, glb_vector = inference(test_loader, model, device, view, data_size, True)
    
    if return_full_metric == False:
        acc = evaluate(labels_vector, target_pred, glb_vector, return_full_metric)
        return acc
    
    nmi, ari, acc, pur, pca_casw, pca_clisi, lda_casw, lda_clisi = evaluate(labels_vector, soft_pred, glb_vector, True)
    if isprint :
        print("Clustering results on soft pred: " + str(labels_vector.shape[0]))
        print('ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f} pca_cASW={:.4f} pca_cLISI={:.4f} lda_cASW={:.4f} lda_cLISI={:.4f}'.format(acc, nmi, ari, pur, pca_casw, pca_clisi, lda_casw, lda_clisi))
    
    nmi, ari, acc, pur, pca_casw, pca_clisi, lda_casw, lda_clisi = evaluate(labels_vector, target_pred, glb_vector, True)
    if isprint :
        print("Clustering results on target pred: " + str(labels_vector.shape[0]))
        print('ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f} pca_cASW={:.4f} pca_cLISI={:.4f} lda_cASW={:.4f} lda_cLISI={:.4f}'.format(acc, nmi, ari, pur, pca_casw, pca_clisi, lda_casw, lda_clisi))
    if return_latent:
        return acc, nmi, pur, ari, pca_casw, pca_clisi, lda_casw, lda_clisi, target_pred, glb_vector
    return acc, nmi, pur, ari, pca_casw, pca_clisi, lda_casw, lda_clisi
