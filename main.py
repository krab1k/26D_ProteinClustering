import numpy as np
import argparse
from time import time
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import dimred
import cluster
import validate


def dimensionality_reduction(data, verbose=False):
    reduced = {'none': data}

    for dim_method in dimred.methods:
        time0 = time()
        X = dim_method(data)
        time1 = time()
        reduced[dim_method.__name__] = X

        if verbose:
            print(f'Dimensionality reduction: {dim_method.__name__} finished in {time1 - time0:.1f} s')

    return reduced


def clustering(datapack, scale=False, verbose=False):
    cluster_labels = {}

    for dr, data in datapack.items():
        for clust_method in cluster.methods:
            if scale:
                data = StandardScaler().fit_transform(data)
            time0 = time()
            labels = clust_method(data)
            time1 = time()

            cluster_labels[(dr, clust_method.__name__)] = labels
            if verbose:
                print(f'Clustering: {clust_method.__name__} finished in {time1 - time0:.1f} s')

    return cluster_labels


def validation(cluster_labels, distances):
    validation_results = {}
    for method_types, labels in cluster_labels.items():
        ratio = validate.validate(labels, distances)
        validation_results[method_types] = ratio

    return validation_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('distances_file')
    args = parser.parse_args()

    data = np.loadtxt(args.input_file, delimiter=',', dtype=np.int_)
    print('Vectors loaded')

    distances = np.loadtxt(args.distances_file, delimiter=',', dtype=np.float32)
    print('Distances loaded')

    reduced = dimensionality_reduction(data, verbose=True)
    clusters = clustering(reduced, verbose=True)
    results = validation(clusters, distances)

    for (dr_method, clust_method), result in results.items():
        labels = clusters[(dr_method, clust_method)]
        if dr_method == 'none':
            for method in dimred.methods:
                X = reduced[method.__name__]
                plt.scatter(X.T[0], X.T[1], c=labels)
                plt.title(f'No DR + {clust_method} + {method.__name__} projection => {result:.2f} sucessful ')
                plt.savefig(f'output/xxx_{clust_method}_{method.__name__}.png')
                plt.clf()
        else:
            X = reduced[dr_method]
            plt.scatter(X.T[0], X.T[1], c=labels)
            plt.title(f'{dr_method} + {clust_method} => {result:.2f} sucessful ')
            plt.savefig(f'output/{dr_method}_{clust_method}.png')
            plt.clf()
