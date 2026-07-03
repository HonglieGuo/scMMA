import scanpy as sc
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def search_res(adata, n_clusters):
    """
    Searching corresponding resolution according to given cluster number
    """
    print('Searching resolution...')
    label = -1
    start=0.1
    end=3.0
    res = 0.5
    eps = 1e-5
    
    while (end - start > eps):
        res = start + (end - start) / 2
        sc.tl.leiden(adata, random_state=66, resolution=res)
        count_unique = adata.obs['leiden'].cat.categories.shape[0]
        print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break
        elif count_unique > n_clusters:
            end = res
        else:
            start = res
    
    assert label==1, "Resolution is not found. Please try bigger range or smaller step!." 
    
    return res


def leiden_clustering(features, n_cluster, resolution=None):
    '''
    resolution: Value of the resolution parameter, use a value above
            (below) 1.0 if you want to obtain a larger (smaller) number
            of communities.
    '''

    print("\nInitializing cluster centroids using the leiden method.")

    adata0 = sc.AnnData(features)
    sc.tl.pca(adata0, svd_solver='arpack', n_comps=50)
    sc.pp.neighbors(adata0, n_neighbors=15) #, use_rep="X"
    if resolution is None:
        resolution = search_res(adata0, n_cluster)

    adata0 = sc.tl.leiden(adata0, resolution=resolution, random_state=66, copy=True)
    y_pred = adata0.obs['leiden']
    y_pred = np.asarray(y_pred, dtype=int)

    features = pd.DataFrame(adata0.X, index=np.arange(0, adata0.shape[0]))
    group = pd.Series(y_pred, index=np.arange(0, adata0.shape[0]), name="Group")
    mergeFeature = pd.concat([features, group], axis=1)

    init_centroids = []
    
    for g in range(n_cluster):
        fs = mergeFeature[mergeFeature['Group'] == g].values
        d = elucidence(fs)
        d = np.sum(d, axis=1) - np.max(d, axis=1)
        print('d.shape', d.shape)
        d = d.reshape(-1)
        init_centroids.append(fs[np.argmin(d)])
    
    
    init_centroid = np.asarray(mergeFeature.groupby("Group").mean())
    return y_pred, init_centroid

def elucidence(X: np.ndarray):  
    # Calculate squared sum of each row  
    sum_square = np.sum(X**2, axis=1, keepdim=True)  # Shape: (n, 1)  
    # print(sum_square)
    
    # Calculate the distance matrix using broadcasting  
    # D = sqrt((x1^2 + x2^2 - 2*x1*x2))  
    distances = sum_square + sum_square.T - 2 * X @ X.T
    mask = np.ones_like(distances) - np.eye(distances.shape[0])
    distances = mask * distances
    return np.sqrt(distances)

def plot_vis_losses(vis_losses, f_name=''):
    """
    Plot the training loss over epochs.

    Parameters:
    vis_losses (list of float): A list containing the total loss values at each epoch.
    """
    # Create a figure and axis
    plt.figure(figsize=(10, 6))

    # Plot the loss curve
    plt.plot(vis_losses, color='blue', linestyle='-', linewidth=2, marker='o', markersize=4)

    # Add titles and labels
    plt.title('Training Loss Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)

    # Display grid lines for better readability
    plt.grid(True, linestyle='--', alpha=0.6)

    # Show the plot
    # plt.show()
    plt.savefig(f'./result/vis_{f_name}_loss.pdf', format='pdf', dpi=300)
    plt.close()

def plot_loss_and_accuracy(total_loss, acc_list, f_name=''):  
    """Plots the loss and accuracy over epochs with improved readability."""  
    
    epochs = range(1, len(total_loss) + 1)  
    n_epochs = len(epochs)  

    fig, ax1 = plt.subplots(figsize=(10, 6))  

    # Plot Loss  
    ax1.set_xlabel('Epochs', fontsize=16)  
    ax1.set_ylabel('Loss', color='tab:red', fontsize=16)  
    ax1.plot(epochs, total_loss, color='tab:red', marker='o', label='Loss',  
              linewidth=3, markersize=8)  
    ax1.tick_params(axis='y', labelcolor='tab:red', labelsize=12)  

    # Secondary y-axis for Accuracy  
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Accuracy', color='tab:blue', fontsize=16)  
    ax2.plot(epochs, acc_list, color='tab:blue', marker='s', label='Accuracy',  
              linewidth=3, markersize=8)  
    ax2.tick_params(axis='y', labelcolor='tab:blue', labelsize=12)  

    # Customize x-axis ticks  
    if n_epochs > 10:  
        interval = max(1, n_epochs // 10)  
        ax1.set_xticks(range(1, n_epochs + 1, interval))  
    else:  
        ax1.set_xticks(epochs)  
    ax1.set_xticklabels(ax1.get_xticks(), rotation=45, ha='right', fontsize=12)  

    # Title and Grid  
    plt.title('Loss and Accuracy over Epochs', fontsize=18, fontweight='bold')  
    # ax1.grid(which='both', linestyle='--', linewidth=1, alpha=0.7)  

    # Legend placement  
    ax1.legend(loc='upper left', fontsize=12)  
    ax2.legend(loc='upper right', fontsize=12)  

    # Adjust layout and save  
    plt.tight_layout()  
    plt.savefig(f'./result/{f_name}_loss_acc.pdf', format='pdf', dpi=300, bbox_inches='tight')  
    plt.close()  
    
def save_latent(glb_vector, dataset):
    np.save(f'./latent/{dataset}_glb_vector.npy', glb_vector)
    np.save(f'./latent/{dataset}_Qs_vector.npy', Qs_vector)