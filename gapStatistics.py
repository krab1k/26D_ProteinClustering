#!/usr/bin/env python -*- coding: utf-8 -*-
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import tqdm
import multiprocessing
import argparse
from sklearn.metrics import calinski_harabaz_score
from sklearn.metrics import silhouette_score

def dispersion(data, k, other_metrics=False):
    k_means_model_ = MiniBatchKMeans(n_clusters=k, max_iter=50, n_init=5).fit(data)
    if other_metrics:
        ch = calinski_harabaz_score(data, k_means_model_.labels_)
        sh = silhouette_score(data, k_means_model_.labels_)
        return np.log(k_means_model_.inertia_), ch, sh
    else:
        return np.log(k_means_model_.inertia_)

def reference_dispersion(data, num_clusters, num_reference_bootstraps):
    dispersions = [dispersion(generate_uniform_points(data), num_clusters) for i in range(num_reference_bootstraps)]
    mean_dispersion = np.mean(dispersions)
    return mean_dispersion

def generate_uniform_points(data):
    mins = np.argmin(data, axis=0)
    maxs = np.argmax(data, axis=0)

    num_dimensions = data.shape[1]
    num_datapoints = data.shape[0]

    reference_data_set = np.zeros((num_datapoints, num_dimensions))
    for i in range(num_datapoints):
        for j in range(num_dimensions):
            reference_data_set[i][j] = random.uniform(data[mins[j]][j], data[maxs[j]][j])

    return reference_data_set

def gap_statistic(data, nth_cluster, num_reference_bootstraps):
    actual_dispersion, ch, sh = dispersion(data, nth_cluster, other_metrics=True)
    ref_dispersion = reference_dispersion(data, nth_cluster, num_reference_bootstraps)
    return actual_dispersion, ref_dispersion, ch, sh

def process(x):
    data, k, n_ref = x
    actual, reference, ch, sh = gap_statistic(data, k, n_ref)
    return reference - actual, ch, sh

def main(filename, n_components):
    np.random.seed(0)
    N = 15000
    MAX_CLUSTERS = 160
    NUM_REFERENCE_BOOTSTRAPS = 10
    
    data = np.load(filename)
    data = data[:N]
    
    pca = PCA(n_components=n_components)
    data = pca.fit_transform(data)

    dispersion_values = np.zeros((MAX_CLUSTERS, 2))

    clusters = list(range(10, MAX_CLUSTERS + 1, 10))
    with multiprocessing.Pool(6) as pool:
        x = [(data, i, NUM_REFERENCE_BOOTSTRAPS) for i in clusters]
        gaps = np.array(list(tqdm.tqdm(pool.imap(process, x), total=len(x))))

    print(gaps)
    print(gaps.shape)
    best = clusters[np.argmax(gaps[:, 0])]
    
    plt.rcParams['figure.figsize'] = [16, 12]
    fig = plt.figure()
    host = fig.add_subplot(111)
    ch_plot = host.twinx()
    sh_plot = host.twinx()
    host.xaxis.set_ticks(clusters)
    host.set_xlabel('Number of clusters')
    host.set_ylabel('Gap score')
    ch_plot.set_ylabel('Calinski Harabaz score')
    sh_plot.set_ylabel('Silhouette score')
    host.set_title(f'{N} data points, PCA to {n_components} dimensions, best gap score for {best} clusters ')
    
    p1 = host.bar(clusters, gaps.T[0], color='red', label='Gap score')
    p2 = ch_plot.bar(np.array(clusters) + 1, gaps.T[1], color='blue', label='Calinski Harabaz score')
    p3 = sh_plot.bar(np.array(clusters) + 2, gaps.T[2], color='green', label='Silhouette score')

    lns = [p1, p2, p3]
    sh_plot.spines['right'].set_position(('outward', 60))
   
    host.yaxis.label.set_color('red')
    ch_plot.yaxis.label.set_color('blue')
    sh_plot.yaxis.label.set_color('green')

    plt.savefig(f'scores_d{n_components}.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str)
    parser.add_argument('n_components', type=int)

    args = parser.parse_args()
    main(args.data, args.n_components)
