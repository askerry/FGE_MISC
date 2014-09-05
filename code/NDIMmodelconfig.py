# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 13:49:27 2014

@author: amyskerry
"""
import sys
sys.path.append('/Users/amyskerry/Dropbox/antools/utilities')
from sklearn import svm, cluster
from sklearn.cross_decomposition import PLSRegression
from sklearn.cluster import AffinityPropagation


#main modeling parameters here
def SVM_model():
    model=svm.SVC(C=.01, kernel='linear') #svm.SVC implements multiclass as one vs. one
    #SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,gamma=0.0, kernel='rbf', max_iter=-1, probability=False, random_state=None,shrinking=True, tol=0.001, verbose=False)
    #model=svm.LinearSVC() #svm.LinearSVC() implements multiclass as one vs. test
    #see http://scikit-learn.org/stable/modules/svm.html
    return model
def PLSR_model(ncomp):
    #PLS regression is particularly suited when the matrix of predictors has more variables than observations, and when there is multicollinearity among X values. By contrast, standard regression will fail in these cases.
    model=PLSRegression(n_components=ncomp, scale=True)
    return model
def KMEANS_model(numclust):
    model= cluster.KMeans(n_clusters=numclust)
    return model
def affinityprop():
    ap=AffinityPropagation(damping=0.8, max_iter=500, affinity='euclidean', convergence_iter=25, copy=True, preference=None)
    return ap
    
#can implement PLSR-DA with PLSR by making dummy columns for each label
    #dummylabel=np.array([[1,1,0,0],[0,0,1,1]]).T
    #myplsda=PLSRegression().fit(X=Xdata,Y=dummylabel)
    #mypred= myplsda.predict(Xdata)