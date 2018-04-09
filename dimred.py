from sklearn import manifold, decomposition

methods = []
n_neighbors = 10
n_components = 2


def dim_red(f):
    methods.append(f)
    return f


@dim_red
def pca(X):
    return decomposition.TruncatedSVD(n_components=n_components).fit_transform(X)


@dim_red
def lle_standard(X):
    return manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                           eigen_solver='dense',
                                           method='standard').fit_transform(X)


@dim_red
def lle_ltsa(X):
    return manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                           eigen_solver='dense',
                                           method='ltsa').fit_transform(X)


@dim_red
def lle_hessian(X):
    return manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                           eigen_solver='dense',
                                           method='hessian').fit_transform(X)


@dim_red
def lle_modified(X):
    return manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                           eigen_solver='dense',
                                           method='modified').fit_transform(X)


@dim_red
def isomap(X):
    return manifold.Isomap(n_neighbors, n_components).fit_transform(X)


@dim_red
def mds(X):
    return manifold.MDS(n_components, max_iter=100, n_init=1).fit_transform(X)


@dim_red
def se(X):
    return manifold.SpectralEmbedding(n_components=n_components,
                                      n_neighbors=n_neighbors).fit_transform(X)


@dim_red
def tsne(X):
    return manifold.TSNE(n_components=n_components, init='pca', random_state=0).fit_transform(X)
