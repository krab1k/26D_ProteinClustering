from sklearn import cluster
from sklearn.neighbors import kneighbors_graph

methods = []
n_neighbors = 10
n_clusters = 10


def cluster_method(f):
    methods.append(f)
    return f


@cluster_method
def mean_shift(X):
    return cluster.MeanShift(bin_seeding=True, n_jobs=1).fit_predict(X)


@cluster_method
def minibatch_kmeans(X):
    return cluster.MiniBatchKMeans(n_clusters=n_clusters).fit_predict(X)


@cluster_method
def spectral(X):
    return cluster.SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack',
                                      affinity="nearest_neighbors").fit_predict(X)


@cluster_method
def aglomerative(X):
    connectivity = kneighbors_graph(X, n_neighbors=n_neighbors, include_self=False)
    connectivity = 0.5 * (connectivity + connectivity.T)
    return cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage="average",
                                           affinity="cityblock", connectivity=connectivity).fit_predict(X)


@cluster_method
def birch(X):
    return cluster.Birch(n_clusters=n_clusters).fit_predict(X)


@cluster_method
def dbscan(X):
    return cluster.DBSCAN().fit_predict(X)
