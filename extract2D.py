from sklearn.manifold import Isomap
import matplotlib.pyplot as plt
import sys
import numpy as np
from collections import Counter
from sklearn import manifold
#from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
from matplotlib.ticker import NullFormatter
from sklearn.neighbors import KNeighborsClassifier



def dbParser(db):
    

    pdbnames, vects = [], []

    fin = open(db) 
    for l in fin:
        l = l.strip().split(',')
        print(len(l))
        pdbnames.append(l[0])
        vects.append(l[1:len(l)])
        #vects.append(l[-1])

    dataDict = dict()
    dataDict['pdbchains_names'] = np.asarray(pdbnames)
    dataDict['vectors'] = np.asarray(vects)
    print(np.asarray(vects))
    return dataDict



#getting datasets
db = dbParser(db='file26Vectors.txt')
print (db)
X, Y = db['vectors'], db['pdbchains_names']
print ('\n X',X.shape)
print ('\n Y',Y.shape)
print(X.dtype)
#reducing dimmension
dmap = manifold.Isomap(n_components=2)
X_t= dmap.fit_transform(X)
print('\n2D', X_t)




    

   

