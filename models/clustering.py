"""=================================================================="""
"""=================== LIBRARIES ===================================="""
"""=================================================================="""


"""========================================================================================================"""
# Top function holder which runs cluster computation of network features, optional preprocessing using PCA,
# assigning clusters to dataloader as new targets and adjusts the network to the number of clusters accordingly.




from tqdm import tqdm, trange
import os
import sys, numpy as np, time
import faiss
import matplotlib.pyplot as plt
def compute_clusters(dataloader, cluster_network, opt):
    image_clusters = generate_cluster_labels(
        dataloader, cluster_network, opt)
    return image_clusters


"""========================================================================================================"""
# Function holder which runs feature computation on training images <compute_features>, optional PCA-preprocessing,
# clustering <cluster> and optional umap generation.


def generate_cluster_labels(dataloader, cluster_network, opt):
    features = compute_features(dataloader, cluster_network, opt)
    image_clusters = cluster(features, opt.num_queries)
    return image_clusters


"""========================================================================================================"""
# Using the network.features of choice, run through the full dataset and compute feature vectors for all
# input images. Collect them and pass them to <feature_preprocessing> and/or <cluster> for clustering.
# If the make_umap_plots-flag is set, the additional label vector is saved.


def compute_features(dataloader, cluster_network, opt):
    cluster_network.eval()
    dataloader.dataset.random_sample = False

    iterator = tqdm(dataloader, position=1)
    iterator.set_description('Computing features & Clustering... ')

    num_samples = len(dataloader)
    print(num_samples)
    feature_coll = np.zeros((num_samples * 49, opt.feature_dim))
    last_idx = 0
    for i, input_image in enumerate(iterator):
        input_image = input_image.to(opt.device)
        # [1, C, H, W] -> [1, C, H*W] -> [1, H*W, C] -> [1*H*W, C]
        feature = cluster_network(input_image).flatten(2).permute(0, 2, 1).flatten(0, 1).detach().cpu().numpy()
        num_feat = feature.shape[0]
        print(feature.shape)
        feature_coll[last_idx : last_idx + num_feat] = feature
        last_idx += num_feat

    return np.vstack(feature_coll)


"""========================================================================================================"""
# Run optional feature preprocessing to select the <pca_dim> most important features for quicker cluster search.


def feature_preprocessing(features, pca_dim=256):
    _, feature_dim = features.shape
    features = features.astype('float32')

    PCA_generator = faiss.PCAMatrix(feature_dim, pca_dim, eigen_power=-0.5)
    PCA_generator.train(features)

    # PCA-Whitening
    assert PCA_generator.is_trained
    features = PCA_generator.apply_py(features)

    # L2 normalization
    row_sum = np.linalg.norm(features, axis=1)
    features = features/row_sum[:, np.newaxis]

    return features


"""========================================================================================================"""
# Cluster the set of feature vectors passed from <compute_features> using faiss.Clustering().
# The number of clusters if given by <num_cluster>. Clusters are measured via L2-distance.
# <image_list> is passed as <image_clusters>  in <compute_clusters_and_set_dataloader_labels> to
# <adjust_dataloader_labels> to be set as new training labels in the provided dataset.


def cluster(features, num_cluster):
    nsample, d = features.shape
    niter = int(20 * (nsample / (1000 * num_cluster))) # As a rule of thumb there is no consistent improvement of the k-means quantizer beyond 20 iterations and 1000 * k training points
    # faiss implementation of k-means
    clus = faiss.Kmeans(d, num_cluster, gpu=True, spherical=True, seed=42, niter = niter)
    clus.max_points_per_centroid = 10000000

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    clus.train(features)
    return clus.centroids