# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 15:16:04 2014

@author: amyskerry
"""
import os
import pandas as pd
import numpy as np
import NDIMmodelconfig as mcfg
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
import datetime
from itertools import cycle
import scipy.stats
from sklearn.decomposition import PCA, FastICA
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d   

###########################################################################################    
#general
########################################################################################### 

def quicksave(obj,filename):
    with open(filename, 'wb') as output:
        pickler = pickle.Pickler(output, pickle.HIGHEST_PROTOCOL)
        pickler.dump(obj)
    
def plotbar(array, xlabel='', ylabel='', title='', xticklabels=None, ax=None, figsize=[4,3], ylim=None):
    if not ax:
        f,ax=plt.subplots(figsize=figsize)
    ax.bar(range(len(array)),array)
    if xticklabels:
        ax.set_xticks(np.arange(len(xticklabels))+.5)
        ax.set_xticklabels(xticklabels, rotation=90)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if ylim:
        ax.set_ylim(ylim)
    
def plotmatrix(matrix, xlabel='', ylabel='', ax=None, title='', colorbar=False, figsize=[4,3], xticklabels=None, yticklabels=None):
    matrix=np.array(matrix)
    if not ax:
        f,ax=plt.subplots(figsize=figsize)
    ax.set_ylim(0,len(matrix))
    ax.set_xlim(0,len(matrix[0]))
    ax.set_title(title)
    im=ax.pcolor(matrix)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xticklabels:
        ax.set_xticks(np.arange(len(xticklabels))+.5)
        ax.set_xticklabels(xticklabels, rotation=90)
    if yticklabels:
        ax.set_yticks(range(len(yticklabels)))
        ax.set_yticklabels(yticklabels)
    if colorbar:
        plt.colorbar(im, cax=ax)
        
def plot3d(matrix, indices=[0,1,2], colors=['b'], dimname='dimension', figsize=[4,3]):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=figsize)
    ax = Axes3D(fig)
    ax.scatter(matrix[:,indices[0]],matrix[:,indices[1]],matrix[:,indices[2]], c=colors, cmap=plt.cm.spectral)
    if type(dimname)==str:
        dimname=[dimname for el in range(3)]
    ax.set_xlabel('%s #%s' % (dimname[0], indices[0]))
    ax.set_ylabel('%s #%s' % (dimname[1], indices[1]))
    ax.set_zlabel('%s #%s' % (dimname[2], indices[2]))
    return ax

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    def draw(self, renderer):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
            FancyArrowPatch.draw(self, renderer)
            
def reversehash(dictobj):
    return {item[1]:item[0] for item in dictobj.items()}

###########################################################################################    
#preprocessing
###########################################################################################    

def createitem2emomapping(stimdf):
    item2emomapping={}
    for index,row in stimdf.iterrows():
        item2emomapping['q%s' % row['qnum']]=row['emotion']
    return item2emomapping
    
def setup(dfile, checks, item2emomapping, orderedemos, appraisals, subjcols, fixeditems, normalize=False,visualize=True):    
    avgthresh=checks.values()[0][0]
    indthresh=checks.values()[0][1]
    df=pd.read_csv(dfile)
    print "starting with %s rows...." %(len(df))
    if 'surprised_dim' in appraisals and 'disgusted_dim' in appraisals:
        print "hack to convert disgusted and surprised from emos to appraisals"
        df.loc[:,'surprised_dim']=df['Surprised_extent']
        df.loc[:,'disgusted_dim']=df['Disgusted_extent']
    df=checkfornonnull(df, appraisals)
    print "reduced to %s items by removing all-null entries...." % len(df)
    df,fcounts=checkrange(df,appraisals)
    print "altered out of range values for %s datapoints....." % fcounts
    df=deletecheckfailsubjs(df, checks.keys()[0], avgthresh)
    print "reduced to %s items by removing check-failing subjects (avgthresh=%s)....." %(len(df), avgthresh)
    df=deletecheckfailitems(df, checks.keys()[0], indthresh)
    print "reduced to %s check-passing items (indthresh=%s)....." %(len(df), indthresh)
    df=checkweird(df, orderedemos)
    print "reduced to %s after eliminating acquisition errors......" % len(df)
    df=defineitems(df, item2emomapping)  
    df,subjdf=makesubjdf(df, subjcols=subjcols)
    print "added item and emo column" 
    df=deletecols(df, ['qemo', 'qlabel', 'main_character', 'submission_date'])
    print "data from %s unique subjects" %(len(subjdf)) 
    df,dfexplicit=splitmainsubj(df, appraisals, orderedemos, fixeditems=fixeditems)
    if normalize:
        df=normalizedf(df, appraisals)
    countitems(df, visualize=visualize)
    return df, subjdf, dfexplicit
    
def normalizedf(df,appraisals):
    df.loc[:,appraisals]=(df[appraisals] - df[appraisals].mean()) / (df[appraisals].std())     #normalize by row
    return df
    
def splitdf(df, subsets, fixedcols):
    dfdict={}
    for key in subsets.keys():
        cols=[el for el in fixedcols]
        cols.extend(subsets[key])
        dfdict[key]=df[cols]
    print "split dataframe into %s dataframes: %s" %(len(subsets.keys()), ', '.join(subsets.keys()))
    return dfdict

def splitmainsubj(df,appraisals, orderedemos, fixeditems=[]):
    emocols,appraisalcols=[el for el in fixeditems], [el for el in fixeditems]
    emocols.extend([emo+'_extent' for emo in orderedemos])
    appraisalcols.extend(appraisals)
    dfexplicit=df[emocols]
    df=df[appraisalcols]
    return df, dfexplicit

def checkfornonnull(df, appraisals):
    dffilter=[]
    for index,row in df.iterrows():
        dffilter.append(all(row[appraisals].notnull()))
    return df[dffilter]
    
def checkrange(df,appraisals):
    df.loc[:,appraisals]=df[appraisals].astype(float)
    failures=0
    for index,row in df.iterrows():
        if any(row[appraisals]>10) or any(row[appraisals]<0):
            for a in appraisals:
                if row[a]<0 or row[a]>10:
                    warnings.warn('value out of range in row %s: column %s has an out of range value %s' %(index, a, row[a]))
                    df.ix[index,a]=5
                    failures += 1
    return df, failures

def deletecheckfailsubjs(df, checkcol, avgthresh):
    keepers=[subj for subj in df['subjid'].unique() if df[checkcol][df['subjid']==subj].mean()>avgthresh]
    df=df[[(subj in keepers) for subj in df['subjid'].values]]
    return df
    
def deletecheckfailitems(df, checkcol, indthresh):
    return df[df[checkcol]>indthresh]
    
def checkweird(df, orderedemos):
    numitems=[]
    numemos=[]
    for index,row in df.iterrows():
        items=set([el for el in row.values if type(el)==str and el[0]=='q'])
        emos=set([el for el in row.values if type(el)==str and el in orderedemos])
        numitems.append(len(items)) #find number of distinct items in row
        numemos.append(len(emos))
    df=df[[(length==1) for length in numitems]]
    df=df[[(length<=2) for length in numemos]] #if there's just one bad column, don't worry
    return df

def defineitems(df, item2emomapping):
    items=[[el for el in row.values if type(el)==str and el[0]=='q'][0] for index,row in df.iterrows()] #find number of distinct items in row
    df.loc[:,'item']=items
    df.loc[:,'emo']=[item2emomapping[item] for item in items]
    cols=['item', 'emo']
    cols.extend([col for col in df.columns if col not in cols])
    return df[cols]

def makesubjdf(df, subjcols=[]):
    allsubjcols=['subjid']
    allsubjcols.extend(subjcols)
    subjdf=pd.DataFrame(index=df['subjid'].unique(),columns=allsubjcols)
    for subj in df['subjid'].unique():
        datarow=[tup[1][allsubjcols] for tup in df.iterrows() if tup[1]['subjid']==subj if any(tup[1][subjcols].notnull())]
        if datarow:
            subjdf.loc[subj,:]=datarow[0]
    df=df[[col for col in df.columns if col not in subjcols]]
    return df, subjdf
            
def deletecols(df, dlist):
    cols=[c for c in df.columns if not all(df[c].isnull()) and not any([(string in c) for string in dlist])]
    print "reduced from %s to %s columns" %(len(df.columns), len(cols)) 
    return df[cols]
    
def countitems(df, visualize=True):
    counts=df.groupby('item')['rownum'].count()
    if visualize:
        plt.figure(figsize=[16,2])
        counts.plot(kind='bar')
        plt.show()
    print "%s to %s responses per item (mean=%.3f)" %(min(counts.values), max(counts.values), np.mean(counts.values))
            
    

###########################################################################################    
#dimensionality reduction
###########################################################################################    
    
def affinitypropcluster(datamatrix, true_labels, axislabels, dim1=0, dim2=1):
    '''takes a dataset and returns and plots set of clusters using affinity propogation (advantage: don't have to speciy K)'''
    apm = mcfg.affinityprop()
    af=apm.fit(datamatrix)
    cluster_centers = af.cluster_centers_indices_
    labels = af.labels_
    real2clustermapping={l:labels[ln] for ln,l in enumerate(true_labels)}
    cluster2realmapping={}
    for l in set(labels):
        name=true_labels[cluster_centers[l]]
        reals=[i[0] for i in real2clustermapping.items() if i[1]==l]
        cluster2realmapping[name]=reals
    n_clusters = len(cluster_centers)
    print('Estimated number of clusters: %d' % n_clusters)
    plt.figure()
    colors = sns.color_palette('husl', n_clusters)
    #for each cluster, find the members and and plot around center
    for k, color in zip(range(n_clusters), colors):
        class_members = labels == k
        nameindex=cluster_centers[k]
        cluster_center, cluster_name = datamatrix[nameindex], true_labels[nameindex]
        plt.plot(datamatrix[class_members, dim1], datamatrix[class_members, dim2], '.', markerfacecolor=color)
        plt.plot(cluster_center[dim1], cluster_center[dim2], 'o', markerfacecolor=color,
                markeredgecolor='k', markersize=14)
        plt.text(cluster_center[dim1]+.01, cluster_center[dim2]+.01, cluster_name, size=10,horizontalalignment='right',verticalalignment='top',
            bbox=dict(facecolor='w',alpha=.6))
        for x in datamatrix[class_members]:
            plt.plot([cluster_center[dim1], x[dim1]], [cluster_center[dim2], x[dim2]], color=color)
    plt.xlabel(axislabels[dim1])
    plt.ylabel(axislabels[dim2])
    plt.title('Estimated number of dimension clusters: %d. \nPlotted in space of emotions %s and %s.' % (n_clusters,axislabels[dim1],axislabels[dim2]))
    plt.show()
    return af, cluster2realmapping
    
def optimalKclusters1(matrix):
    from scipy.cluster.vq import kmeans
    matrix=np.array(matrix)
    K = range(1,len(matrix)+1) 
    KM = [kmeans(matrix,k) for k in K]
    avgWithinSS = [var for (cent,var) in KM] # mean within-cluster sum of squares
    WSSdiffs=[avgWithinSS[eln-1]-el for eln,el in enumerate(avgWithinSS) if eln>0]
    belowZero=[True if el<=0 else False for el in WSSdiffs]
    try:
        optimalK=belowZero.index(True)
    except:
        optimalK=WSSdiffs.index(min(WSSdiffs))-1
    # plot elbow curve
    f,ax = plt.subplots()
    ax.plot(K, avgWithinSS, 'b*-')
    ax.plot(K[optimalK], avgWithinSS[optimalK], marker='o', markersize=12, 
    markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
    plt.xlabel('Number of clusters')
    plt.ylabel('Average within-cluster sum of squares')
    plt.title('Elbow for KMeans clustering')
    plt.show()
    return optimalK
    
def kmeansclustering(matrix, dimlabels, axislabels, numclust=0):
    if numclust==0:
        numclust=len(axislabels)
    k_means = mcfg.KMEANS_model(numclust)
    k_means.fit(matrix)
    kclusters=k_means.labels_
    kclusters, dimlabels = zip(*sorted(zip(kclusters, dimlabels))) # zipping together, sorting, and unzipping
    indclusters=list(set(kclusters))
    clusterdict={k:[] for k in indclusters}
    for cn, c in enumerate(kclusters):
        clusterdict[c].append(dimlabels[cn])
    return clusterdict, {'k_means':k_means, 'kclusters':kclusters, 'itememos':dimlabels, 'numclusters':numclust}
    
def hierarchicalcluster(datamatrix, dimlabels, similarity='euclidean', colorthresh='default'):
    '''plots dendrogram and returns clustering (item-1 x 4 array. first two columns are indices of clusters, 3rd column = distance between those clusters, 4th column = # of
      original observations in the cluster) and dend (dictionary of the data structures computed to render the
      dendrogram). see api here: http://hcluster.damianeads.com/cluster.html'''
    import hcluster
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clustering=hcluster.linkage(datamatrix, metric=similarity)
        if colorthresh=='default':
            color_threshold=0.7*max(clustering[:,2]) #all descendents below a cluster node k will be assigned the same color if k is the first node below color_threshold. links connecting nodes with distances >= color_threshold are colored blue. default= 0.7*max(clustering[:,2])
        else: 
            color_threshold=colorthresh*max(clustering[:,2])
        fig = plt.figure()
        dend=hcluster.dendrogram(clustering, labels=dimlabels, leaf_rotation=90, color_threshold=color_threshold)
        plt.tight_layout()
    return clustering, dend
    
def reduction(matrix, ncomp=None, reductiontype='PCA'):
    #set up figure
    f,ax=plt.subplots(1,3, figsize=[10,3])
    if reductiontype=='PCA':
        ca=PCA()
    elif reductiontype=='ICA':
        ca=FastICA()
    if ncomp:
        ca.n_components=ncomp
    #do basic reduction
    obsscores=ca.fit_transform(matrix)#scores of our observations on the 3 components
    dimloadings=ca.components_ #loadings of initial data dimensions onto these components (eigenvectors)
    recoveredinput=ca.inverse_transform(obsscores) #recomputed obs x dimension matrix based on scores on the components, and weights on components
    if ncomp: #if we know our intended # of components
        plotmatrix(obsscores, xlabel='components', ylabel='observations',ax=ax[0], title='obsscores')
        plotmatrix(dimloadings, xlabel='initial dimensions', ylabel='components',ax=ax[1], title='dim loadings')
    else: #if we are fitting an initial model to evaluate dimensions
        plotmatrix(matrix, xlabel='initial dimensions', ylabel='initial observations', ax=ax[0], title='initial input')
        if reductiontype=='PCA':
            plotscree(ca, ax=ax[1])
    plotmatrix(recoveredinput, xlabel='initial dimensions', ylabel='transformed observations',ax=ax[2], title='recovered input (all components)')
    plt.tight_layout()
    print "reduced to %s dimensions" % (len(ca.components_))
    if reductiontype=='PCA':
        print "explaining %.3f%% of variance" %(100*np.sum(ca.explained_variance_ratio_))
    return obsscores, dimloadings, recoveredinput
    
def plotscree(ca, ax=0):
    if ax==0:
        f,ax=plt.subplots()
    else:
        ax.plot(ca.explained_variance_ratio_)
        ax.set_title('scree plot')
        ax.set_ylabel('variance explained')
        ax.set_xlabel('eigenvector')
    
def plotcomponentcorrs(obsscores, ax=None, label=''):
    #sns.corrplot(pd.DataFrame(obsscores), method='pearson', annot=False, ax=ax)
    correlations=pd.DataFrame(obsscores).corr(method='pearson').values
    corrs=correlations[np.tril_indices(len(correlations),1)]
    plotmatrix(np.tril(correlations, -1), colorbar=False, figsize=[3,2], ax=ax, xlabel=label)
    plt.title('correlation of components in raw input space\n mean corr= %.6f' %(np.mean(corrs)))
    plt.tight_layout()

def plotcomponentsininputspace(matrix, axislabels, item2emomapping, dimloadings, dimindices=[0,1,2], dimlabels=[], componentindices=[0], title=None):#plot observations on 3 components
    #plot observations on 3 components
    axisdims=list(set(item2emomapping.values()))
    colornums=sns.color_palette('husl', len(axisdims))
    labelcolors={el:colornums[eln] for eln,el in enumerate(axisdims)}
    colors=[labelcolors[item2emomapping[el]] for el in axislabels]
    if len(dimlabels)>0:
        dimname=[dimlabels[el] for el in dimindices]
    else:
        dimname='input dim'
    ax=plot3d(matrix,indices=dimindices, colors=colors, dimname=dimname)
    meanx,meany,meanz=np.mean(matrix[:,dimindices[0]]),np.mean(matrix[:,dimindices[1]]),np.mean(matrix[:,dimindices[2]])
    for x in componentindices:    
        comp=dimloadings[x]
        cx,cy,cz=comp[dimindices[0]],comp[dimindices[1]],comp[dimindices[2]]
        ax.plot([meanx], [meany], [meanz], '.', markersize=5, color='b')  
        print cx,cy,cz
        a = Arrow3D([meanx,cx],[meany,cy],[meanz,cz], mutation_scale=10, lw=.5, arrowstyle="-|>")
        ax.add_artist(a)
    plt.show()
    
def plotinputincomponentspace(obsscores, axislabels, item2emomapping, componentindices=[1,2,3], title=None):
    #plot observations on 3 components
    axisdims=list(set(item2emomapping.values()))
    colornums=sns.color_palette('husl', len(axisdims))
    labelcolors={el:colornums[eln] for eln,el in enumerate(axisdims)}
    colors=[labelcolors[item2emomapping[el]] for el in axislabels]
    plot3d(obsscores,indices=componentindices, colors=colors, dimname='component')
            
###########################################################################################    
#modeling
###########################################################################################    
            
def createCVindices(df, classcol, generalizationcol=None, nfolds=2):
    '''create cross-validation indices as new column "CVI" in the dataset. if generalizationcol is provided, all instances of a given value of that column will be assigned to a single fold, requiring generalization across this dimension'''
    df.loc[:,'CVI']=[np.nan for el in range(len(df))]
    for u in df[classcol].unique(): #for each value of classification label
        tempdf=df[df[classcol]==u]
        if not generalizationcol: #not requiring generalization across any dimension
            cvindices=np.hstack([np.zeros(len(tempdf)/2+np.mod(len(tempdf),2)),np.ones(len(tempdf)/2)])
            np.random.shuffle(cvindices)
            df.ix[tempdf.index.values,'CVI']=cvindices
        else: # requiring generalization across the <withinfoldcol> dimension
            generalizationuniques=tempdf[generalizationcol].unique()
            cvindices=np.hstack([np.zeros(len(generalizationuniques)/2+np.mod(len(generalizationuniques),2)),np.ones(len(generalizationuniques)/2)])
            np.random.shuffle(cvindices) 
            for gn,g in enumerate(generalizationuniques):
                innertempdf=tempdf[tempdf[generalizationcol]==g] #assign same CV index to all entries that g
                df.ix[innertempdf.index.values,'CVI']=cvindices[gn]
    return df
    
def compressdf(df, labelcol, exemplarcol):
    '''take df and compress to summary df with single array of features (average) for each unique exemplar'''
    item2indexdict=df.groupby(exemplarcol).groups #create dict to map exemplars to labels
    modeldata=df.groupby(exemplarcol).mean()
    modeldata.insert(0, exemplarcol, modeldata.index)
    modeldata.insert(1, labelcol, [df.ix[item2indexdict[i][0],labelcol] for i in modeldata.index.values])
    return modeldata

def classify(df, labelcol, featurenames, folds, exemplarcol, gencol=None):
    '''classify with linear SVM'''
    df=createCVindices(df, labelcol, generalizationcol=gencol, nfolds=folds['nfolds'])
    train=df[df['CVI']==folds['train']]
    test=df[df['CVI']==folds['test']]
    train=compressdf(train, labelcol, exemplarcol)
    test=compressdf(test, labelcol, exemplarcol)
    if gencol and any([v in train[exemplarcol].values for v in test[exemplarcol].values]):
        warnings.warn('% repeated across training and test folds (supposed to be generalizing across %s)' %(gencol))
    train_X,test_X=train[featurenames].values,test[featurenames].values
    train_Y,test_Y=train[labelcol].values,test[labelcol].values
    clf=mcfg.SVM_model()
    clf=clf.fit(train_X,train_Y)
    predictions=clf.predict(test_X)
    return predictions, test_Y, test[exemplarcol].values

def updateexemplars(df, exemplarcol, iteration, exemplarindices, exemplaraccs):
    '''add accuracies for individual items from this iteration of classification'''
    colname='acc_iter%s' % iteration
    df.loc[:,colname]=np.nan
    uniqueexemplars=df[exemplarcol].values
    for e in uniqueexemplars:
        try:
            relitems=[a for an,a in enumerate(exemplaraccs) if exemplarindices[an]==e]
            exemplarmean=np.mean(relitems)
        except:
            exemplarmean=np.nan
        df.loc[df[df[exemplarcol]==e].index,colname]=exemplarmean
    return df

def compressexemplars(df):
    '''average the individual item accuracy across different iterations of the iterative classifiction procedure'''
    acccols=[col for col in df.columns if 'acc_iter' in col]
    df.loc[:,'mean_acc']=df[acccols].mean(axis=1, skipna=True)
    for col in acccols:
        del df[col]
    return df
    
def iterativeclassification(df, classcol, featurecols, folds, labelorder, exemplarcol=None, gencol=None, iterations=10):
    confmat,meanacc,labelaccs,accs,sims=[],[],[],[],[]
    uniqueexemplars=list(set(df[exemplarcol]))
    exemplardf=pd.DataFrame(index=range(len(uniqueexemplars)))
    exemplardf.loc[:,exemplarcol]=uniqueexemplars
    for i in range(iterations):
        predictions, labels, exemplarindices=classify(df,classcol, featurecols, folds, exemplarcol=exemplarcol, gencol=gencol)
        confmat.append(makeconfmat(predictions, labels, labelorder))
        m, ec, exemplaracc=accuracies(labels,predictions,labelorder)
        meanacc.append(m)
        labelaccs.append(ec)
        exemplardf=updateexemplars(exemplardf, exemplarcol, i, exemplarindices, exemplaracc)
    confmat=np.nanmean(confmat, axis=0)
    meanacc=np.nanmean(meanacc)
    labelaccs=np.nanmean(labelaccs, axis=0)
    exemplardf=compressexemplars(exemplardf)
    print "overall classification across iterations: %.2f%%" %(meanacc*100)
    return confmat, meanacc, labelaccs, exemplardf

def similarity(array1,array2, simtype='pearsonr'):
    '''compute similarity between 2 arrays'''
    if simtype=='pearsonr':
        corr,p=scipy.stats.pearsonr(array1,array2)
    if simtype=='spearmanr':
        corr,p=scipy.stats.spearmanr(array1,array2)
    return corr,p
    
def compareconfmats(confvalues,labelorder,comparisons, plot=False, ax=None):
    '''compare raw behavioral confusion matrix from NDE with with confusion matrix from a model-based classification'''
    compconf=comparisons['cprop']
    compconf=compconf.ix[labelorder,labelorder]
    comparray,confarray=compconf.values.flatten(),confvalues.flatten()
    if any([any(compconf.columns!=labelorder), any(compconf.index!=labelorder)]):
        warnings.warn('label values are not aligned')
    corr,p=similarity(confarray,comparray, simtype='spearmanr')
    if plot:
        tempdf=pd.DataFrame(data={'behavioral confusions (NDE)':comparray, 'model confusions':confarray})
        ax.set_title('spearmanr=%.3f, p=%.3f' %(corr,p))
        sns.regplot(x='behavioral confusions (NDE)', y='model confusions', data=tempdf,ax=ax)
    return corr,p
    
def makeconfmat(predictions, labels, labelorder):
    '''make a confusion matrix from a set of predictions and labels'''
    confmat=np.zeros([len(labelorder),len(labelorder)])
    pairs=zip(labels,predictions)
    for label in labelorder:
        tups=[tup for tup in pairs if tup[0]==label]
        for t in tups:
            confmat[labelorder.index(label),labelorder.index(t[1])]+=1
    return confmat
    
def accuracies(labels, predictions, labelorder):
    '''returns trial-wise accuracy (acc), mean accuracy across all trials in the iteration (meancc) and label-wise accuracy (labelaccs)'''
    acc=[int(pred==labels[predn]) for predn,pred in enumerate(predictions)]
    meanacc=np.mean(acc)
    labelaccs=[]
    for label in labelorder:
        relguesses=[a for an, a in enumerate(acc) if labels[an]==label]
        labelaccs.append(np.mean(relguesses))
    return meanacc, labelaccs, acc

def visualizeclassification(confmat, labelaccs, orderedlabels, figsize=[12,3], comparisons=None):
    '''plot classification results'''
    if comparisons:
        f,ax=plt.subplots(1,4, figsize=figsize)
    else:
        f,ax=plt.subplots(1,2, figsize=figsize)
    plotmatrix(confmat, title='confusion matrix', ylabel='label', xlabel='guesses', xticklabels=orderedlabels,yticklabels=orderedlabels, ax=ax[0])
    plotbar(labelaccs, xlabel='true label', ylabel='percent correct (%)', title='accuracies', xticklabels=orderedlabels, ax=ax[1], ylim=[0,1])
    if comparisons:
        #compare emoaccs
        axis=ax[2]
        rawvals=[comparisons['csummary'].accuracy.ix[emo] for emo in orderedlabels]
        tempdf=pd.DataFrame(data={'raw accuracy':rawvals, 'model accuracy':[el for el in labelaccs]})
        r,p=scipy.stats.pearsonr(tempdf['raw accuracy'],tempdf['model accuracy'])
        axis.set_title('pearsonr=%.3f, p=%.3f' %(r,p))
        sns.regplot(x='raw accuracy', y='model accuracy', data=tempdf,ax=axis)
        #compare to confusions from NDE
        axis=ax[3]
        compareconfmats(confmat,orderedlabels,comparisons, plot=True, ax=axis)
        
class ClassificationResult():
    def __init__(self, model, labelcol, features, gencol, iterations, orderedlabels):
        self.model=model
        self.labelcol=labelcol
        self.features=features
        self.gencol=gencol
        self.iterations=iterations
        self.orderedlabels=orderedlabels
        self.confmat=None
        self.meanacc=None
        self.labelaccs=None
        self.exemplaraccs=None
    def save(self, filename):
        with open(filename, 'wb') as output:
            pickler = pickle.Pickler(output, pickle.HIGHEST_PROTOCOL)
            pickler.dump(self)

def classificationwrapper(modeldata,allresults, chosenmodel, labelcol, features, gencol, iterations, orderedlabels, folds, rootdir, comparisons=None, visualize=True):
    '''cleaning up the notebook with a dummy wrapper'''
    result= ClassificationResult(chosenmodel, labelcol, features, gencol, iterations, orderedlabels)
    print "classifying %s with generalization across %s: %s iterations of randomized split-half train/test" %(chosenmodel, gencol,iterations)
    result.confmat, result.meanacc, result.labelaccs, result.exemplaraccs=iterativeclassification(modeldata, labelcol, features, folds, orderedlabels, exemplarcol='item', gencol=gencol, iterations=iterations)
    allresults['%s_%s' %(chosenmodel, gencol)]=result
    if visualize:    
        visualizeclassification(result.confmat, result.labelaccs, orderedlabels, comparisons=comparisons)
        plt.tight_layout()
        plt.show()
    result.save(os.path.join(rootdir, 'results','%s_%s_results.pkl' %(chosenmodel, gencol)))
    return allresults
    
###########################################################################################    
#comparing
###########################################################################################    

def plotacccomparison(resultsdict, keys=[], flags=[], benchmark=None):
    '''plot meanacc for each model space'''
    if len(keys)==0:
        keys=list(set([k[0:k.index('_')] for k in resultsdict.keys()]))
    if len(flags)==0:
        flags=list(set([k[0:k.index('_')] for k in resultsdict.keys()]))
    f,ax=plt.subplots(1,len(flags), sharey=True)
    f.suptitle('model comparisons')
    ax[0].set_ylabel('accuracy (mean % across iterations)')
    for fn,f in enumerate(flags):
        if f:
            xlabel="generalization across %s" % f
        else:
            xlabel="no generalization"
        accs=[resultsdict[k+'_'+f].meanacc for k in keys]
        axis=ax[fn]
        axis.bar(range(len(accs)),accs)
        axis.set_xticks(np.arange(len(keys))+.5)
        axis.set_xticklabels(keys, rotation=90)
        axis.set_xlabel(xlabel)
        if benchmark:
            axis.plot(axis.get_xlim(),[benchmark, benchmark], linestyle='--', alpha=.4)
    
###########################################################################################    
#similarity spaces
###########################################################################################    

def createfeaturespace(df, features):
    meandf=df[features].groupby('item').mean()
    labels=meandf.index.values
    matrix=meandf.values
    return {'itemavgs':list(matrix), 'itemlabels':list(labels)}
    
def createsimmtxfromconfmat(confmat, labels):
    return {'confmat':confmat, 'labels':labels}   