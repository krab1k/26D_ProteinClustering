import argparse
import os
from RCCobject import RCC
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from multiprocessing import Pool
import matplotlib.pyplot as plt

def create_rcc(pdb, chain):
    return RCC(pdb, chain).RCCvector


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clustering of protein chains')
    parser.add_argument('pdb_directory')
    parser.add_argument('list_of_chains')

    args = parser.parse_args()

    ids = []
    with open(args.list_of_chains) as chains_f:
        for line in chains_f:
            pdb_id, chain_id = line.strip().split(':')
            ids.append((pdb_id, chain_id))

    with Pool(1) as pool:
        data = [(os.path.join(args.pdb_directory, f'{pdb_id.lower()}.pdb'), chain_id) for pdb_id, chain_id in ids[:10]]
        rccs = pool.starmap(create_rcc, data)

    arr = np.array(rccs)
    pca = PCA(n_components=2)
    fitted = pca.fit_transform(arr)
    kmeans = KMeans(n_clusters=3).fit(fitted)
    plt.scatter(fitted.T[0], fitted.T[1], c=kmeans.labels_)
    plt.show()
