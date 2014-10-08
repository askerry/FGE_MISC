# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 15:16:04 2014

@author: amyskerry
"""
import os
import sys

sys.path.append('/mindhive/saxelab/scripts/aesscripts/')
import pandas as pd
import numpy as np
from FGE_MISC.code.config import mcfg, cfg
import mypymvpa.utilities.stats as mus
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
from copy import deepcopy
import scipy.stats
import collections
import FGE_MISC.code.vizfuncs as viz
from copy import deepcopy

# ##########################################################################################
#misc
########################################################################################### 

def quicksave(obj, filename):
    with open(filename, 'wb') as output:
        pickler = pickle.Pickler(output, pickle.HIGHEST_PROTOCOL)
        pickler.dump(obj)


def quickload(filename):
    '''load obj from filename'''
    with open(filename, 'r') as inputfile:
        obj = pickle.load(inputfile)
    return obj


def nonnancorr(array1, array2):
    if np.isnan(np.sum(array1)) or np.isnan(np.sum(array2)):
        array1, array2 = np.array(array1), np.array(array2)
        arrays = np.array(zip(array1, array2))
        nonnans = np.where(~np.isnan(arrays))[0]
        nonnans = [item[0] for item in collections.Counter(nonnans).items() if item[1] == 2]
        arrays = arrays[nonnans]
        array1, array2 = zip(*arrays)
    r, p = scipy.stats.pearsonr(array1, array2)
    return r, p, len(array1)


def reversehash(dictobj):
    '''reverses a dict with 1 to 1 mapping'''
    return {item[1]: item[0] for item in dictobj.items()}


###########################################################################################
#preprocessing
###########################################################################################    

def createitem2emomapping(stimdf):
    item2emomapping = {}
    for index, row in stimdf.iterrows():
        item2emomapping['q%s' % row['qnum']] = row['emotion']
    return item2emomapping


def setup(dfile, checks, item2emomapping, orderedemos, appraisals, subjcols, fixeditems, normalize=False,
          visualize=True):
    avgthresh = checks.values()[0][0]
    indthresh = checks.values()[0][1]
    df = pd.read_csv(dfile)
    if 'safey' in df.columns:
        df=df.rename(columns={'safey':'safety', 'safey_qemo':'safety_qemo', 'safey_qlabel':'safety_qemo'})
    print "starting with %s rows...." % (len(df))
    if 'surprised_dim' in appraisals and 'disgusted_dim' in appraisals:
        print "hack to convert disgusted and surprised from emos to appraisals"
        df.loc[:, 'surprised_dim'] = df['Surprised_extent']
        df.loc[:, 'disgusted_dim'] = df['Disgusted_extent']
    df = checkfornonnull(df, appraisals)
    print "reduced to %s items by removing all-null entries...." % len(df)
    df, fcounts = checkrange(df, appraisals)
    print "altered out of range values for %s datapoints....." % fcounts
    df = deletecheckfailsubjs(df, checks.keys()[0], avgthresh)
    print "reduced to %s items by removing check-failing subjects (avgthresh=%s)....." % (len(df), avgthresh)
    df = deletecheckfailitems(df, checks.keys()[0], indthresh)
    print "reduced to %s check-passing items (indthresh=%s)....." % (len(df), indthresh)
    df = checkweird(df, orderedemos)
    print "reduced to %s after eliminating acquisition errors......" % len(df)
    df = defineitems(df, item2emomapping)
    df, subjdf = makesubjdf(df, subjcols=subjcols)
    print "added item and emo column"
    df = deletecols(df, ['qemo', 'qlabel', 'main_character', 'submission_date'])
    print "data from %s unique subjects" % (len(subjdf))
    df, dfexplicit = splitmainsubj(df, appraisals, orderedemos, fixeditems=fixeditems)
    if normalize:
        df = normalizedf(df, appraisals)
    countitems(df, visualize=visualize)
    return df, subjdf, dfexplicit


def normalizedf(df, appraisals):
    df.loc[:, appraisals] = (df[appraisals] - df[appraisals].mean()) / (df[appraisals].std())  #normalize by row
    return df


def splitdf(df, subsets, fixedcols):
    dfdict = {}
    for key in subsets.keys():
        cols = [el for el in fixedcols]
        cols.extend(subsets[key])
        dfdict[key] = df[cols]
    print "split dataframe into %s dataframes: %s" % (len(subsets.keys()), ', '.join(subsets.keys()))
    return dfdict


def splitmainsubj(df, appraisals, orderedemos, fixeditems=[]):
    emocols, appraisalcols = [el for el in fixeditems], [el for el in fixeditems]
    emocols.extend([emo + '_extent' for emo in orderedemos])
    appraisalcols.extend(appraisals)
    dfexplicit = df[emocols]
    df = df[appraisalcols]
    return df, dfexplicit


def checkfornonnull(df, appraisals):
    dffilter = []
    for index, row in df.iterrows():
        dffilter.append(all(row[appraisals].notnull()))
    return df[dffilter]


def checkrange(df, appraisals):
    df.loc[:, appraisals] = df[appraisals].astype(float)
    failures = 0
    for index, row in df.iterrows():
        if any(row[appraisals] > 10) or any(row[appraisals] < 0):
            for a in appraisals:
                if row[a] < 0 or row[a] > 10:
                    warnings.warn(
                        'value out of range in row %s: column %s has an out of range value %s' % (index, a, row[a]))
                    df.ix[index, a] = 5
                    failures += 1
    return df, failures


def deletecheckfailsubjs(df, checkcol, avgthresh):
    keepers = [subj for subj in df['subjid'].unique() if df[checkcol][df['subjid'] == subj].mean() > avgthresh]
    df = df[[(subj in keepers) for subj in df['subjid'].values]]
    return df


def deletecheckfailitems(df, checkcol, indthresh):
    return df[df[checkcol] > indthresh]


def checkweird(df, orderedemos):
    numitems = []
    numemos = []
    for index, row in df.iterrows():
        items = set([el for el in row.values if type(el) == str and el[0] == 'q'])
        emos = set([el for el in row.values if type(el) == str and el in orderedemos])
        numitems.append(len(items))  #find number of distinct items in row
        numemos.append(len(emos))
    df = df[[(length == 1) for length in numitems]]
    df = df[[(length <= 2) for length in numemos]]  #if there's just one bad column, don't worry
    return df


def defineitems(df, item2emomapping):
    items = [[el for el in row.values if type(el) == str and el[0] == 'q'][0] for index, row in
             df.iterrows()]  #find number of distinct items in row
    df.loc[:, 'item'] = items
    df.loc[:, 'emo'] = [item2emomapping[item] for item in items]
    cols = ['item', 'emo']
    cols.extend([col for col in df.columns if col not in cols])
    return df[cols]


def makesubjdf(df, subjcols=[]):
    allsubjcols = ['subjid']
    allsubjcols.extend(subjcols)
    subjdf = pd.DataFrame(index=df['subjid'].unique(), columns=allsubjcols)
    for subj in df['subjid'].unique():
        datarow = [tup[1][allsubjcols] for tup in df.iterrows() if tup[1]['subjid'] == subj if
                   any(tup[1][subjcols].notnull())]
        if datarow:
            subjdf.loc[subj, :] = datarow[0]
    df = df[[col for col in df.columns if col not in subjcols]]
    subjdf=subjdf[subjdf.gender.notnull()]
    return df, subjdf

def summarize(df1, subjdf1, df2, subjdf2):
    totaldf=pd.concat([df1, df2])
    tsubjdf=pd.concat([subjdf1,subjdf2])
    print "%s individual item responses, %s subjects" %(len(totaldf), len(tsubjdf))
    def nanorfloat(val):
        try:x=float(val);return True
        except: return False
    ages=tsubjdf.age[[nanorfloat(val) for val in tsubjdf.age.values]]
    ages=[float(el) for el in ages.dropna().values]
    genders=[f.lower() for f in tsubjdf.gender.dropna().values]
    females=len([g for g in genders if g[0]=='f'])
    print "%s females, age: mean(sem)=%.2f(%.2f)" %(females, np.mean(ages), np.std(ages)/np.sqrt(len(ages)))
    counts=countitems(totaldf, visualize=False)

def deletecols(df, dlist):
    cols = [c for c in df.columns if not all(df[c].isnull()) and not any([(string in c) for string in dlist])]
    print "reduced from %s to %s columns" % (len(df.columns), len(cols))
    return df[cols]


def countitems(df, visualize=True):
    counts = df.groupby('item')['rownum'].count()
    if visualize:
        plt.figure(figsize=[16, 2])
        counts.plot(kind='bar')
        plt.show()
    print "%s to %s responses per item (mean=%.3f, sem=%.3f)" % (min(counts.values), max(counts.values), np.mean(counts.values), np.std(counts.values)/np.sqrt(len(counts.values)))
    return counts.values

###########################################################################################    
#dimensionality reduction
###########################################################################################
def regress(X,y):
    from sklearn.linear_model import LinearRegression
    clf= LinearRegression()
    clf.fit(X, y)
    R2=clf.score(X, y)
    predicted_y=clf.predict(X)
    residual_error=y-predicted_y
    return R2, residual_error, predicted_y

def iterativeregression(avgs, rootdir, plotit=False):
    features=avgs.columns
    keptfeatures,varexplained_full,varexplained_ind, iterationpredictions=[],[],[],[]
    initialdata=avgs[features].values
    y=avgs[features].values # y starts as initialdata and subsequently becomes the residuals
    for i in range(len(features)):
        R2s=[]
        for f in range(len(features)): #regress each isolated feature against the data
            if f not in keptfeatures: #(if the feature has not already been selected)
                X = np.array([item[f] for item in initialdata]).reshape(len(y),1)
                R2, residual_error, predy = regress(X,y)
                R2s.append(R2)
            else:
                R2s.append(0)
        fn=R2s.index(np.max(R2s)) # best feature
        keptfeatures.append(fn)
        #recompute the relevant X vector and get its inidividual varexplained (in the fulldataset)
        keeperX = np.array([item[fn] for item in initialdata]).reshape(len(y),1)
        R2, residual_error, predy = regress(keeperX,initialdata)
        varexplained_ind.append(R2)
        #generate full X (of all kept features) to get the residuals
        fullX = np.array([item[keptfeatures] for item in initialdata])
        R2, residual_error, predy = regress(fullX,initialdata)
        iterationpredictions.append(predy)
        y = residual_error # the residuals are now our outcome variable for the subsequent iteration
        varexplained_full.append(R2)
    resultingfeatures=[features[fn] for fn in keptfeatures]
    print (', ').join(resultingfeatures)
    if plotit:
        f,ax=plt.subplots(1,2, figsize=[14,4])
        ax[0].plot(range(len(varexplained_full)), varexplained_full)
        ax[0].set_xlabel('features')
        ax[0].set_ylabel('total variance explained (cumulative)')
        ax[0].set_xticks(range(len(resultingfeatures)))
        ax[0].set_xticklabels(resultingfeatures, rotation=90)
        ax[1].plot(range(len(varexplained_ind)), varexplained_ind)
        ax[1].set_xlabel('features')
        ax[1].set_ylabel('individual feature R-squared')
        ax[1].set_xticks(range(len(resultingfeatures)))
        ax[1].set_xticklabels(resultingfeatures, rotation=90)
        sns.despine()
    results={'df':avgs, 'predictions':iterationpredictions, 'features':resultingfeatures, 'varexp_full':varexplained_full, 'varexp_ind':varexplained_ind}
    quicksave(results, os.path.join(rootdir,'results','ITERATIVERESULTS.pkl'))
    return results


def affinitypropcluster(datamatrix, true_labels, axislabels, dim1=0, dim2=1):
    '''takes a dataset and returns and plots set of clusters using affinity propogation (advantage: don't have to speciy K)'''
    apm = mcfg.affinityprop()
    af = apm.fit(datamatrix)
    cluster_centers = af.cluster_centers_indices_
    labels = af.labels_
    real2clustermapping = {l: labels[ln] for ln, l in enumerate(true_labels)}
    cluster2realmapping = {}
    for l in set(labels):
        name = true_labels[cluster_centers[l]]
        reals = [i[0] for i in real2clustermapping.items() if i[1] == l]
        cluster2realmapping[name] = reals
    n_clusters = len(cluster_centers)
    print('Estimated number of clusters: %d' % n_clusters)
    plt.figure()
    colors = sns.color_palette('husl', n_clusters)
    #for each cluster, find the members and and plot around center
    for k, color in zip(range(n_clusters), colors):
        class_members = labels == k
        nameindex = cluster_centers[k]
        cluster_center, cluster_name = datamatrix[nameindex], true_labels[nameindex]
        plt.plot(datamatrix[class_members, dim1], datamatrix[class_members, dim2], '.', markerfacecolor=color)
        plt.plot(cluster_center[dim1], cluster_center[dim2], 'o', markerfacecolor=color,
                 markeredgecolor='k', markersize=14)
        plt.text(cluster_center[dim1] + .01, cluster_center[dim2] + .01, cluster_name, size=10,
                 horizontalalignment='right', verticalalignment='top',
                 bbox=dict(facecolor='w', alpha=.6))
        for x in datamatrix[class_members]:
            plt.plot([cluster_center[dim1], x[dim1]], [cluster_center[dim2], x[dim2]], color=color)
    plt.xlabel(axislabels[dim1])
    plt.ylabel(axislabels[dim2])
    plt.title('Estimated number of dimension clusters: %d. \nPlotted in space of axislabels %s and %s.' % (
        n_clusters, axislabels[dim1], axislabels[dim2]))
    plt.show()
    return af, cluster2realmapping


def optimalKclusters1(matrix):
    from scipy.cluster.vq import kmeans

    matrix = np.array(matrix)
    K = range(1, len(matrix) + 1)
    KM = [kmeans(matrix, k) for k in K]
    avgWithinSS = [var for (cent, var) in KM]  # mean within-cluster sum of squares
    WSSdiffs = [avgWithinSS[eln - 1] - el for eln, el in enumerate(avgWithinSS) if eln > 0]
    belowZero = [True if el <= 0 else False for el in WSSdiffs]
    try:
        optimalK = belowZero.index(True)
    except:
        optimalK = WSSdiffs.index(min(WSSdiffs)) - 1
        # plot elbow curve
    f, ax = plt.subplots()
    ax.plot(K, avgWithinSS, 'b*-')
    ax.plot(K[optimalK], avgWithinSS[optimalK], marker='o', markersize=12,
            markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
    plt.xlabel('Number of clusters')
    plt.ylabel('Average within-cluster sum of squares')
    plt.title('Elbow for KMeans clustering')
    plt.show()
    return optimalK


def kmeansclustering(matrix, dimlabels, axislabels, numclust=0):
    if numclust == 0:
        numclust = len(axislabels)
    k_means = mcfg.KMEANS_model(numclust)
    k_means.fit(matrix)
    kclusters = k_means.labels_
    kclusters, dimlabels = zip(*sorted(zip(kclusters, dimlabels)))  # zipping together, sorting, and unzipping
    indclusters = list(set(kclusters))
    clusterdict = {k: [] for k in indclusters}
    for cn, c in enumerate(kclusters):
        clusterdict[c].append(dimlabels[cn])
    return clusterdict, {'k_means': k_means, 'kclusters': kclusters, 'itememos': dimlabels, 'numclusters': numclust}


def hierarchicalcluster(datamatrix, dimlabels, similarity='euclidean', colorthresh='default'):
    '''plots dendrogram and returns clustering (item-1 x 4 array. first two columns are indices of clusters, 3rd column = distance between those clusters, 4th column = # of
      original observations in the cluster) and dend (dictionary of the data structures computed to render the
      dendrogram). see api here: http://hcluster.damianeads.com/cluster.html'''
    import hcluster

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clustering = hcluster.linkage(datamatrix, metric=similarity)
        if colorthresh == 'default':
            color_threshold = 0.7 * max(clustering[:,
                                        2])  #all descendents below a cluster node k will be assigned the same color if k is the first node below color_threshold. links connecting nodes with distances >= color_threshold are colored blue. default= 0.7*max(clustering[:,2])
        else:
            color_threshold = colorthresh * max(clustering[:, 2])
        fig = plt.figure()
        dend = hcluster.dendrogram(clustering, labels=dimlabels, leaf_rotation=90, color_threshold=color_threshold)
        plt.tight_layout()
    return clustering, dend


def reduction(matrix, ncomp=None, reductiontype='PCA', plotit=True):
    #set up figure
    if reductiontype == 'PCA':
        ca = mcfg.PCA()
    elif reductiontype == 'ICA':
        ca = mcfg.ICA()
    if ncomp:
        ca.n_components = ncomp
        #do basic reduction
    obsscores = ca.fit_transform(matrix)  #scores of our observations on the 3 components
    dimloadings = ca.components_  #loadings of initial data dimensions onto these components (eigenvectors)
    recoveredinput = ca.inverse_transform(
        obsscores)  #recomputed obs x dimension matrix based on scores on the components, and weights on components
    if plotit:
        f, ax = plt.subplots(1, 3, figsize=[10, 3])
        if ncomp:  #if we know our intended # of components
            viz.plotmatrix(obsscores, xlabel='components', ylabel='observations', ax=ax[0], title='obsscores')
            viz.plotmatrix(dimloadings, xlabel='initial dimensions', ylabel='components', ax=ax[1], title='dim loadings')
        else:  #if we are fitting an initial model to evaluate dimensions
            viz.plotmatrix(matrix, xlabel='initial dimensions', ylabel='initial observations', ax=ax[0],
                       title='initial input')
            if reductiontype == 'PCA':
                viz.plotscree(ca.explained_variance_ratio_, ax=ax[1])
        if not ncomp:
            ncomp='all'
        viz.plotmatrix(recoveredinput, xlabel='initial dimensions', ylabel='transformed observations', ax=ax[2],
                   title='recovered input (%s components)' %(ncomp))
        plt.tight_layout()
    print "reduced to %s dimensions" % (len(ca.components_))
    if reductiontype == 'PCA':
        print "explaining %.3f%% of variance" % (100 * np.sum(ca.explained_variance_ratio_))
        varexplained = ca.explained_variance_ratio_
    else:
        varexplained = None
    return obsscores, dimloadings, recoveredinput, varexplained

def printhighlowloaders(dimloadings, varexplained, dimlabels):
    '''identify high and low loading dimensions for each component'''
    sorteddims=[list(line) for line in list(dimlabels[np.argsort(dimloadings)])]
    for pcn,pc in enumerate(dimloadings):
        s=np.sort(pc)
        varex=varexplained[pcn]
        d=sorteddims[pcn]
        print "component %s (explaining %.2f%% of variance):" %(pcn, varex*100)
        print "    high loaders= %s, %s, %s (%.1f,%.1f,%.1f)"  %(d[-1],d[-2],d[-3],s[-1],s[-2],s[-3])
        #print "    low loaders= %s, %s, %s (%.1f,%.1f,%.1f)"  %(d[1],d[2],d[3],s[1],s[2],s[3])

###########################################################################################
#modeling
###########################################################################################    

def createCVindices(df, classcol, generalizationcol=None, nfolds=2):
    '''create cross-validation indices as new column "CVI" in the dataset. if generalizationcol is provided, all instances of a given value of that column will be assigned to a single fold, requiring generalization across this dimension'''
    df.loc[:, 'CVI'] = [np.nan for el in range(len(df))]
    for u in df[classcol].unique():  #for each value of classification label
        tempdf = df[df[classcol] == u]
        if not generalizationcol:  #not requiring generalization across any dimension
            cvindices = np.hstack([np.zeros(len(tempdf) / 2 + np.mod(len(tempdf), 2)), np.ones(len(tempdf) / 2)])
            np.random.shuffle(cvindices)
            df.ix[tempdf.index.values, 'CVI'] = cvindices
        else:  # requiring generalization across the <withinfoldcol> dimension
            generalizationuniques = tempdf[generalizationcol].unique()
            cvindices = np.hstack([np.zeros(len(generalizationuniques) / 2 + np.mod(len(generalizationuniques), 2)),
                                   np.ones(len(generalizationuniques) / 2)])
            np.random.shuffle(cvindices)
            for gn, g in enumerate(generalizationuniques):
                innertempdf = tempdf[tempdf[generalizationcol] == g]  #assign same CV index to all entries that g
                df.ix[innertempdf.index.values, 'CVI'] = cvindices[gn]
    return df


def compressdf(df, labelcol, exemplarcol):
    '''take df and compress to summary df with single array of features (average) for each unique exemplar'''
    item2indexdict = df.groupby(exemplarcol).groups  #create dict to map exemplars to labels
    modeldata = df.groupby(exemplarcol).mean()
    modeldata.insert(0, exemplarcol, modeldata.index)
    modeldata.insert(1, labelcol, [df.ix[item2indexdict[i][0], labelcol] for i in modeldata.index.values])
    return modeldata


def classify(df, labelcol, featurenames, folds, exemplarcol, gencol=None):
    '''classify with linear SVM'''
    df = createCVindices(df, labelcol, generalizationcol=gencol, nfolds=folds['nfolds'])
    train = df[df['CVI'] == folds['train']]
    test = df[df['CVI'] == folds['test']]
    train = compressdf(train, labelcol, exemplarcol)
    test = compressdf(test, labelcol, exemplarcol)
    if gencol and any([v in train[exemplarcol].values for v in test[exemplarcol].values]):
        warnings.warn('% repeated across training and test folds (supposed to be generalizing across %s)' % (gencol))
    train_X, test_X = train[featurenames].values, test[featurenames].values
    train_Y, test_Y = train[labelcol].values, test[labelcol].values
    clf = mcfg.SVM_model()
    clf = clf.fit(train_X, train_Y)
    predictions = clf.predict(test_X)
    return predictions, test_Y, test[exemplarcol].values


def updateexemplars(df, exemplarcol, iteration, exemplarindices, exemplaraccs):
    '''add accuracies for individual items from this iteration of classification'''
    colname = 'acc_iter%s' % iteration
    df.loc[:, colname] = np.nan
    uniqueexemplars = df[exemplarcol].values
    for e in uniqueexemplars:
        try:
            relitems = [a for an, a in enumerate(exemplaraccs) if exemplarindices[an] == e]
            exemplarmean = np.mean(relitems)
        except:
            exemplarmean = np.nan
        df.loc[df[df[exemplarcol] == e].index, colname] = exemplarmean
    return df


def compressexemplars(df):
    '''average the individual item accuracy across different iterations of the iterative classifiction procedure'''
    acccols = [col for col in df.columns if 'acc_iter' in col]
    df.loc[:, 'mean_acc'] = df[acccols].mean(axis=1, skipna=True)
    for col in acccols:
        del df[col]
    return df


def iterativeclassification(df, classcol, featurecols, folds, labelorder, exemplarcol=None, gencol=None, iterations=10):
    confmat, meanacc, labelaccs, accs, sims = [], [], [], [], []
    uniqueexemplars = list(set(df[exemplarcol]))
    exemplardf = pd.DataFrame(index=range(len(uniqueexemplars)))
    exemplardf.loc[:, exemplarcol] = uniqueexemplars
    for i in range(iterations):
        predictions, labels, exemplarindices = classify(df, classcol, featurecols, folds, exemplarcol=exemplarcol,
                                                        gencol=gencol)
        confmat.append(makeconfmat(predictions, labels, labelorder))
        m, ec, exemplaracc = accuracies(labels, predictions, labelorder)
        meanacc.append(m)
        labelaccs.append(ec)
        exemplardf = updateexemplars(exemplardf, exemplarcol, i, exemplarindices, exemplaracc)
    confmat = np.nanmean(confmat, axis=0)
    confmat=confmat/np.sum(confmat,axis=1)#transform from raw values to proportions
    meanacc = np.nanmean(meanacc)
    labelaccs = np.nanmean(labelaccs, axis=0)
    exemplardf = compressexemplars(exemplardf)
    print "overall classification across iterations: %.2f%%" % (meanacc * 100)
    return confmat, meanacc, labelaccs, exemplardf


def makeconfmat(predictions, labels, labelorder):
    '''make a confusion matrix from a set of predictions and labels'''
    confmat = np.zeros([len(labelorder), len(labelorder)])
    pairs = zip(labels, predictions)
    for label in labelorder:
        tups = [tup for tup in pairs if tup[0] == label]
        for t in tups:
            confmat[labelorder.index(label), labelorder.index(t[1])] += 1
    return confmat


def accuracies(labels, predictions, labelorder):
    '''returns trial-wise accuracy (acc), mean accuracy across all trials in the iteration (meancc) and label-wise accuracy (labelaccs)'''
    acc = [int(pred == labels[predn]) for predn, pred in enumerate(predictions)]
    meanacc = np.mean(acc)
    labelaccs = []
    for label in labelorder:
        relguesses = [a for an, a in enumerate(acc) if labels[an] == label]
        labelaccs.append(np.mean(relguesses))
    return meanacc, labelaccs, acc

def similarity(array1, array2, simtype='pearsonr'):
    '''compute similarity between 2 arrays'''
    if simtype == 'pearsonr':
        corr, p = scipy.stats.pearsonr(array1, array2)
    if simtype == 'spearmanr':
        corr, p = scipy.stats.spearmanr(array1, array2)
    if simtype == 'kendallstau':
        corr, p = mus.kendallstau(array1, array2, type='a', symmetrical=False)
    return corr, p

def compareconfmats(mconf, labelorder, comparisons, simtype, dropdiag=False, subset=None):
    '''compare raw behavioral confusion matrix from NDE with with confusion matrix from a model-based classification'''
    bconf = comparisons['cprop']
    bconf = bconf.ix[labelorder, labelorder]
    if subset:
        modelconfdf=pd.DataFrame(columns=labelorder, index=labelorder, data=mconf)
        labelorder=[el for el in labelorder if el in subset]
        bconf=bconf.ix[subset,subset]
        mconf=modelconfdf.ix[subset,subset].values
    if dropdiag:
        tc=deepcopy(bconf.values)
        tc[np.diag_indices(len(tc[0]))]=np.nan
        barray=tc.flatten()[~np.isnan(tc.flatten())]
        tc=deepcopy(mconf)
        tc[np.diag_indices(len(tc[0]))]=np.nan
        marray=tc.flatten()[~np.isnan(tc.flatten())]
    else:
        barray, marray = bconf.values.flatten(), mconf.flatten()
    if any(bconf.columns != labelorder) or any(bconf.index != labelorder):
        warnings.warn('label values are not aligned')
    corr, p = similarity(marray, barray, simtype=simtype)
    tempdf = pd.DataFrame(data={'behavioral confusions (NDE)': barray, 'model confusions': marray})
    return {'df':tempdf, 'corr':corr, 'p':p, 'simtype':simtype}

def comparemodel2behavior(result, behavior, orderedlabels, simtype='pearsonr', valencedict=cfg.valencedict):
    confmat, labelaccs = result.confmat, result.labelaccs
    rawvals = [behavior['csummary'].accuracy.ix[emo] for emo in orderedlabels]
    #compare accuracies (emo-wise)
    tempdf = pd.DataFrame(data={'raw accuracy': rawvals, 'model accuracy': [el for el in labelaccs]})
    corr,p = similarity(tempdf['raw accuracy'].values, tempdf['model accuracy'].values, simtype=simtype)
    emowise={'df':tempdf, 'corr':corr, 'p':p, 'simtype':simtype}
    #compare accuracies (item-wise)
    ndeitems, ndeaccs = behavior['summary']['item'].values, behavior['summary']['accuracy'].values
    modelitems, modelaccs = result.exemplaraccs['item'].values, result.exemplaraccs['mean_acc'].values
    tempdf = pd.DataFrame(index=ndeitems, data={'raw accuracy': ndeaccs})
    tempdf.ix[modelitems, 'model accuracy'] = modelaccs
    corr,p = similarity(tempdf['raw accuracy'].values, tempdf['model accuracy'].values, simtype=simtype)
    itemwise={'df':tempdf, 'corr':corr, 'p':p, 'simtype':simtype}
    #compare to confusions from NDE
    confmatcomp=compareconfmats(confmat, orderedlabels, behavior, simtype)
    #compare to confusions from NDE without diagonal (i.e. excluding correct answers)
    confmatcomp_nodiag=compareconfmats(confmat, orderedlabels, behavior, simtype, dropdiag=True)
    if valencedict:
        subsets, subsets_nodiag={'simtype':simtype},{'simtype':simtype}
        for key in valencedict.keys():
            subsets[key]=compareconfmats(confmat, orderedlabels, behavior, simtype, subset=valencedict[key])
            subsets_nodiag[key]=compareconfmats(confmat, orderedlabels, behavior, simtype, subset=valencedict[key],dropdiag=True)
    else:
        subsets,subsets_nodiag=None,None
    return {'emowise':emowise, 'itemwise':itemwise, 'confmatcomp':confmatcomp, 'confmatcomp_nodiag': confmatcomp_nodiag, 'confmatcomp_subsets':subsets,'confmatcomp_nodiag_subsets':subsets_nodiag}



class ClassificationResult():
    def __init__(self, model, labelcol, features, gencol, iterations, orderedlabels):
        self.model = model
        self.labelcol = labelcol
        self.features = features
        self.gencol = gencol
        self.iterations = iterations
        self.orderedlabels = orderedlabels
        self.confmat = None
        self.meanacc = None
        self.labelaccs = None
        self.exemplaraccs = None

    def save(self, filename):
        with open(filename, 'wb') as output:
            pickler = pickle.Pickler(output, pickle.HIGHEST_PROTOCOL)
            pickler.dump(self)

def classpar(chosenmodel, modeltypes, allresults, inputs, orderedemos, rootdir):
    modeldata,features=modeltypes[chosenmodel][0],modeltypes[chosenmodel][1]
    #### model without generalization
    rNone=classificationwrapper(modeldata,allresults, chosenmodel, inputs['labelcol'], features, None, inputs['iterations'], orderedemos, inputs['folds'], rootdir)
    #### model with generalization across items
    rItem=classificationwrapper(modeldata,allresults, chosenmodel, inputs['labelcol'], features, 'item', inputs['iterations'], orderedemos, inputs['folds'], rootdir)
    confmat=makeconfdict(rItem, orderedemos)
    return rNone, rItem, confmat

'''
def classificationwrapper(modeldata, allresults, chosenmodel, labelcol, features, gencol, iterations, orderedlabels,
                          folds, rootdir):
    #cleaning up the notebook with a dummy wrapper
    result = ClassificationResult(chosenmodel, labelcol, features, gencol, iterations, orderedlabels)
    print "classifying %s across %s: %s iterations randomized split-half train/test" % (
        chosenmodel, gencol, iterations)
    result.confmat, result.meanacc, result.labelaccs, result.exemplaraccs = iterativeclassification(modeldata, labelcol,
                                                                                                    features, folds,
                                                                                                    orderedlabels,
                                                                                                    exemplarcol='item',
                                                                                                    gencol=gencol,
                                                                                                    iterations=iterations)
    allresults['%s_result_%s' % (chosenmodel, gencol)] = result
    result.save(os.path.join(rootdir, 'results', '%s_%s_results.pkl' % (chosenmodel, gencol)))
    return allresults
'''
def classificationwrapper(modeldata, allresults, chosenmodel, labelcol, features, gencol, iterations, orderedlabels,
                          folds, rootdir):
    '''cleaning up the notebook with a dummy wrapper'''
    result = ClassificationResult(chosenmodel, labelcol, features, gencol, iterations, orderedlabels)
    print "classifying %s with across %s: %s iterations randomized split-1/2 train/test" % (
        chosenmodel, gencol, iterations)
    result.confmat, result.meanacc, result.labelaccs, result.exemplaraccs = iterativeclassification(modeldata, labelcol,
                                                                                                    features, folds,
                                                                                                    orderedlabels,
                                                                                                    exemplarcol='item',
                                                                                                    gencol=gencol,
                                                                                                    iterations=iterations)
    result.save(os.path.join(rootdir, 'results', '%s_%s_results.pkl' % (chosenmodel, gencol)))
    return result
###################################
#individualized predictions
###################################

def createsinglemodel(df, labelcol, featurenames, exemplarcol):
    '''create single trained model based on all data in df'''
    train = compressdf(df, labelcol, exemplarcol)
    train_X, train_Y = train[featurenames].values,train[labelcol].values
    clf = mcfg.SVM_model()
    clf = clf.fit(train_X, train_Y)
    return clf

def explicitacc(df):
    emocols=[col for col in df.columns if 'extent' in col]
    newdata={'item':df['item'].values, 'emo':df['emo'].values, 'rownum':df['rownum'].values, 'maxemos':[], 'pass':[]}
    for index,row in df.iterrows():
        rowmax=max(row[emocols])
        maxemos=[col[:col.index('_')] for col in df.columns if row[col]==rowmax]
        newdata['pass'].append(row['emo'] in maxemos)
        newdata['maxemos'].append(maxemos)
    return pd.DataFrame(index=df.index.values, data=newdata)

def sortcorrects(modeldata, explicitdf):
    eaccdf=explicitacc(explicitdf)
    passers, failers=eaccdf[eaccdf['pass']==True]['rownum'].values, eaccdf[eaccdf['pass']==False]['rownum'].values
    correctdf=modeldata[[m in passers for m in modeldata['rownum'].values]]
    incorrectdf=modeldata[[m in failers for m in modeldata['rownum'].values]]
    return eaccdf, correctdf, incorrectdf

def testfailers(correctdf, incorrectdf, labelcol, features, orderedlabels):
    clf=createsinglemodel(correctdf, labelcol, features, 'item')
    test = compressdf(incorrectdf, 'emo', 'item')
    test_X, test_Y = test[features].values,test[labelcol].values
    predictions = clf.predict(test_X)
    acc=float(sum([int(p==test_Y[pn]) for pn,p in enumerate(predictions)]))/len(predictions)
    return test_Y, predictions, acc

def testinds(eaccdf, correctdf, incorrectdf, labelcol, features, orderedlabels):
    singleguessers= eaccdf[[len(emos)==1 for emos in eaccdf['maxemos'].values]]['rownum'].values
    guessdict={row['rownum']:row['maxemos'][0] for index,row in eaccdf[[len(emos)==1 for emos in eaccdf['maxemos'].values]].iterrows()}
    idf=incorrectdf[[m in singleguessers for m in incorrectdf['rownum'].values]]
    clf=createsinglemodel(correctdf, labelcol, features, 'item')
    itest_X=idf[features]
    itest_intendedemos=idf['emo'].values
    itest_useremos=[guessdict[rownum] for rownum in idf.rownum.values]
    predictions = clf.predict(itest_X)
    useremopercent=float(sum([int(p==itest_useremos[pn]) for pn,p in enumerate(predictions)]))/len(predictions)
    intendedemopercent=float(sum([int(p==itest_intendedemos[pn]) for pn,p in enumerate(predictions)]))/len(predictions)
    return useremopercent, intendedemopercent, idf

def individualizedmodeling(modeldata, explicitdf, chosenmodel, labelcol, features, gencol, iterations, orderedlabels, folds, rootdir):
    eaccdf, correctdf, incorrectdf= sortcorrects(modeldata, explicitdf)
    regtest_Y, regpreds, regacc= testfailers(correctdf, incorrectdf, labelcol, features, orderedlabels)
    useremopercent, intendedemopercent, idf=testinds(eaccdf, correctdf, incorrectdf, labelcol, features, orderedlabels)
    print "training on %s pass items, testing on %s failed items:" %(len(correctdf), len(idf))
    print "accuracy (relative to intended emotion): %.3f, user-consistency (prediction aligns with user max): %.3f" %(intendedemopercent, useremopercent)
    return {'useremoacc':useremopercent, 'intendedemoacc':intendedemopercent, 'numitems':len(idf)}


###########################################################################################    
#similarity spaces
###########################################################################################    

def createfeaturespace(df, features):
    labels = df.item.values
    features = [el for el in features if el not in ('emo', 'CVI', 'item')]
    matrix = df[features].values
    return {'itemavgs': list(matrix), 'itemlabels': list(labels), 'features': features}


def makeconfdict(savedconf, orderedemos):
    labelorder = savedconf.orderedlabels
    oldmatrix = savedconf.confmat
    matrix, labels = [], []
    for e in orderedemos:
        matrix.append(oldmatrix[labelorder.index(e)])
    return {'simmat_across': matrix, 'simmat': None, 'labels': orderedemos}


def makesimspaces(rsadict, item2emomapping, orderedemos, similarity='euclidean', groupcol='emo', itemindices=None,
                  iterations=10):
    '''make RDMs for each of the models.'''
    matrix, items, emos = rsadict['itemavgs'], rsadict['itemlabels'], [item2emomapping[i] for i in
                                                                       rsadict['itemlabels']]
    data = {'item': items, 'emo': emos}
    for fn, f in enumerate(matrix[0]):
        data[fn] = [line[fn] for line in matrix]
    df = pd.DataFrame(data=data)
    grouped, order = compressrdmfeatures(df, groupcol, orderedemos, item2emomapping)
    RSAmat_untransformed = scipy.spatial.distance.cdist(grouped.values, grouped.values, similarity)
    RSAmat = transformsimilarities(RSAmat_untransformed, similarity)
    crossRSAmat = makesimmatrixacross(df, groupcol, order, similarity, item2emomapping, iterations)
    return {'simmat': RSAmat, 'simmat_across': crossRSAmat, 'labels': order}


def compressrdmfeatures(df, groupcol, orderedemos, item2emomapping):
    '''compresses and orders the feature space'''
    appraisals = [col for col in df.columns if col not in ['item', 'emo']]
    grouped = df.groupby(groupcol).mean()
    if groupcol == 'item':
        itemlabels = [el for el in grouped.index.values]
        order = orderitems(itemlabels, orderedemos, item2emomapping)
    elif groupcol == 'emo':
        order = orderedemos
    grouped = grouped.ix[order, appraisals]
    return grouped, order


def makesimmatrixacross(df, groupcol, order, similarity, item2emomapping, iterations):
    '''computes model matrix across different subsets of the data'''
    RSAmats = []
    for i in range(iterations):
        idx1, idx2 = [], []
        for o in order:
            indices = [i for i in df.index if df.ix[i, groupcol] == o]
            np.random.shuffle(indices)
            idx1.extend(indices[0:len(indices) / 2])
            idx2.extend(indices[len(indices) / 2:])
        df1 = df.ix[idx1, :]
        df2 = df.ix[idx2, :]
        grouped1, order1 = compressrdmfeatures(df1, groupcol, order, item2emomapping)
        grouped2, order2 = compressrdmfeatures(df2, groupcol, order, item2emomapping)
        RSAmats.append(scipy.spatial.distance.cdist(grouped1.values, grouped2.values, similarity))
    RSAmat_untransformed = np.mean(RSAmats, axis=0)
    return transformsimilarities(RSAmat_untransformed, similarity)


def transformsimilarities(repdismat, distance):
    '''return transformed similarity matrices'''
    rdm = deepcopy(repdismat)
    if distance == 'euclidean':
        rdm = transformedm(rdm)
    elif distance == 'pearsonr':
        rdm = np.arctanh(rdm)
    else:
        warnings.warn('transformation not implemented for distance type %s' % distance)
    return rdm


def orderitems(itemlabels, orderedemos, item2emomapping):
    '''order items first based on orderedemos, then based on item number'''
    ordereditems = []
    for e in orderedemos:
        ordereditems.extend(
            ['q%.0f' % el for el in np.sort([int(item[1:]) for item in itemlabels if item2emomapping[item] == e])])
    return ordereditems


def euclideandistancematrix(matrix):
    '''take a matrix of items in rows and features in columns and return a similarity space of items'''
    distancematrix = scipy.spatial.distance.cdist(matrix.values, matrix.values, 'euclidean')
    return distancematrix, transformedm(distancematrix)


def transformedm(dm):
    '''transforms a euclidean distance matrix by subtracting the min and dividing by max-min'''
    dm = np.array(dm).astype(float)
    maxdist, mindist = np.max(dm), np.min(dm)
    tdm = (dm - mindist) / (maxdist - mindist)
    return tdm


#specialized similarity matrix functions    
def makeNDEconfmat(conf, labelorder):
    conf = conf.ix[labelorder, labelorder]
    return {'simmat_across': -1 * np.array(conf.values), 'simmat': None, 'labels': conf.columns}


def makeconfRDM(flag, chosenmodel, confmatsaves):
    cfdict = confmatsaves['%s_confs_%s' % (chosenmodel, flag)]
    matrix = -1 * np.array(cfdict['simmat_across'])
    return {'simmat_across': list(matrix), 'labels': cfdict['labels']}
