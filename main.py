import numpy as np
import argparse
from time import time
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import dimred
import cluster
import validate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('distances_file')
    args = parser.parse_args()

    data = np.loadtxt(args.input_file, delimiter=',', dtype=np.int_)
    print('Vectors loaded')

    distances = np.loadtxt(args.distances_file, delimiter=',', dtype=np.float32)
    print('Distances loaded')

    for dim_method in dimred.methods:
        time0 = time()
        X = dim_method(data)
        time1 = time()
        print(f'Dimensionality reduction: {dim_method.__name__} finished in {time1 - time0:.1f} s')
        for clust_method in cluster.methods:
            X = StandardScaler().fit_transform(X)
            time0 = time()
            labels = clust_method(X)
            time1 = time()

            ratio = validate.validate(labels, distances)
            print(f'Clustering: {clust_method.__name__} finished in {time1 - time0:.1f} s')

            plt.scatter(X.T[0], X.T[1], c=labels)
            plt.title(f'{dim_method.__name__} + {clust_method.__name__} => {ratio:.2f} sucessful')
            plt.show()


