# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 09:48:26 2014

@author: amyskerry
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import pandas as pd
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d, Axes3D
import collections
import seaborn as sns
import warnings
import FGE_MISC.code.config as cfg

# #general

try:
    sns.set_style('white')
    sns.set_context('notebook')
except:
    sns.set_axes_style("white", "notebook")


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


def similarity(array1, array2, simtype='pearsonr'):
    '''compute similarity between 2 arrays'''
    if simtype == 'pearsonr':
        corr, p = scipy.stats.pearsonr(array1, array2)
    if simtype == 'spearmanr':
        corr, p = scipy.stats.spearmanr(array1, array2)
    return corr, p


def tickify(ticklabels):
    for v in cfg.tickcfg.values():
        try:
            nticklabels = [v[el] for el in nticklabels]
            return nticklabels
        except:
            pass
    return ticklabels

#############################
#NDE
#############################

def plotconfusions(NDE_output, orderedemos):
    f, ax = plt.subplots(1, 2, figsize=[10, 4])
    heatmapdf(NDE_output['rawconfusions'], xlabel='guessed emotion', ylabel='item', title='raw confusions',
              xticklabels=orderedemos, ax=ax[0])
    heatmapdf(NDE_output['propconfusions'], xlabel='guessed emotion', ylabel='item', title='proportions',
              xticklabels=orderedemos, ax=ax[1])


def plotaccuracy(NDE_output):
    f = plt.figure(figsize=[16, 2])
    NDE_output['summary']['accuracy'].plot(kind='bar')
    plt.show()


def plotcollapsedresults(NDE_output, orderedemos):
    f, ax = plt.subplots(1, 3, figsize=[18, 4])
    heatmapdf(NDE_output['craw'], xlabel='guessed emotion', ylabel='item', title='raw confusions',
              xticklabels=orderedemos, yticklabels=orderedemos, ax=ax[0])
    heatmapdf(NDE_output['cprop'], xlabel='guessed emotion', ylabel='item', title='proportions',
              xticklabels=orderedemos, yticklabels=orderedemos, ax=ax[1])
    ax[2].clear()
    NDE_output['csummary']['accuracy'].plot(kind='bar', yerr=NDE_output['csummary']['SEM'], ax=ax[2])
    ax[2].set_ylabel('accuracy(+/- SEM across items)')


def heatmapdf(df, xlabel=None, ylabel=None, title=None, xticklabels=None, yticklabels=None, ax=None):
    if ax:
        pass
    else:
        f, ax = plt.subplots()
    ax.pcolor(df)
    if xticklabels:
        ax.set_xticks(np.arange(len(df.columns)) + .5)
        ax.set_xticklabels(df.columns, rotation=90)
        ax.set_xlim([0, len(df.columns)])
    if yticklabels:
        ax.set_yticks(np.arange(len(df.index)))
        ax.set_yticklabels(df.index)
        ax.set_ylim([0, len(df.index)])
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    plt.colorbar(plt.pcolor(df), ax=ax)


#############################   
#NDIM
#############################

def plotbar(array, yerr=None, xlabel='', ylabel='', title='', xticklabels=None, ax=None, fmt=None, figsize=[4, 3],
            ylim=None, colors=None):
    if not ax:
        f, ax = plt.subplots(figsize=figsize)
    if yerr:
        ax.errorbar(range(len(array)), array, yerr=yerr, color=colors, fmt=fmt)
    else:
        ax.bar(range(len(array)), array, color=colors)
    if xticklabels:
        ax.set_xticks(np.arange(len(xticklabels)) + .5)
        ax.set_xticklabels(xticklabels, rotation=90)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if ylim:
        ax.set_ylim(ylim)
    return ax


def simplebar(values, width=.9, bars=True, elinewidth=2, markersize=None, fmt=None, xtickrotation=None, figsize=[4, 4],
              yerr=None, xlabel=None, xlim=None, xticklabels=None, ylabel=None, yticklabels=None, title=None, ylim=None,
              colors=['b'], chanceline=None, despine=False, show=True):
    '''takes various optional parameters and makes simple bar chart. at minumim requires set of values to plot, nothing else.
    example of more complex (and strange) usage: simplebar(values, yerr=[.2, .4, .5, .3, .6, .1], xlabel="x", ylabel="y", yticklabels=['1','2','6','7', 'g', 'h',  'j', 'v', '4'], xticklabels=['a','b','c','d','e','f'], chanceline={'chance':3, 'linetype':'solid', 'linecolor':'r'}, ylim=[-3,10])'''
    #deal with some sns version issues
    try:
        sns.set_style('white')
        sns.set_context('notebook')
    except:
        sns.set_axes_style("white", "notebook")
    f, ax = plt.subplots(figsize=figsize)
    if bars:
        ax.bar(range(len(values)), values, color=colors, width=width)
    if yerr:
        if len(colors) > 1:
            while len(colors) < len(values):
                colors.extend(colors)
            for vn, v in enumerate(values):
                baseval = [0 for el in values]
                basesem = [0 for el in values]
                baseval[vn] = v
                basesem[vn] = yerr[vn]
                ax.errorbar(np.arange(len(baseval)) + width / 2, baseval, yerr=basesem, color=colors[vn],
                            ecolor=colors[vn], linestyle='None', fmt=fmt, elinewidth=elinewidth, markersize=markersize)
        else:
            ax.errorbar(np.arange(len(values)) + width / 2, values, yerr=yerr, color=colors, linestyle='None', fmt=fmt,
                        elinewidth=elinewidth, markersize=markersize)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if xticklabels:
        if xtickrotation != None:
            rotation = xtickrotation
        else:
            if len(xticklabels[0]) > 3:
                rotation = 90
            else:
                rotation = 0
        ax.set_xticks(np.arange(len(xticklabels)) + width / 2)
        ax.set_xticklabels(xticklabels, rotation=rotation)
    else:
        ax.set_xticklabels([])
    if yticklabels:
        ax.set_yticklabels(yticklabels)
    if ylim:
        ax.set_ylim(ylim)
    if not xlim:
        xlim = [-width / 8, len(values)]
    ax.set_xlim(xlim)
    if chanceline:
        chance = chanceline['chance']
        try:
            linetype = chanceline['linetype']
        except:
            linetype = 'dashed'
        try:
            linecolor = chanceline['linecolor']
        except:
            linecolor = 'orange'
        ax.plot(xlim, [chance for el in xlim], ls=linetype, color=linecolor)
    if despine:
        sns.despine(fig=f, ax=ax, top=True, right=True, left=False, bottom=False, trim=False)
    if show:
        plt.show()
    return ax


def plotmatrix(matrix, xlabel='', ylabel='', ax=None, title='', colorbar=False, figsize=[4, 3], xticklabels=[],
               yticklabels=[]):
    matrix = np.array(matrix)
    if not ax:
        f, ax = plt.subplots(figsize=figsize)
    ax.set_ylim(0, len(matrix))
    ax.set_xlim(0, len(matrix[0]))
    ax.set_title(title)
    im = ax.pcolor(matrix)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(xticklabels) > 0:
        ax.set_xticks(np.arange(len(xticklabels)) + .5)
        ax.set_xticklabels(xticklabels, rotation=90)
    if len(yticklabels) > 0:
        ax.set_yticks(range(len(yticklabels)))
        ax.set_yticklabels(yticklabels)
    if colorbar:
        plt.colorbar(im)
        #plt.colorbar(im, cax=ax)


def compareconfmats(confvalues, labelorder, comparisons, plot=False, ax=None):
    '''compare raw behavioral confusion matrix from NDE with with confusion matrix from a model-based classification'''
    compconf = comparisons['cprop']
    compconf = compconf.ix[labelorder, labelorder]
    comparray, confarray = compconf.values.flatten(), confvalues.flatten()
    if any([any(compconf.columns != labelorder), any(compconf.index != labelorder)]):
        warnings.warn('label values are not aligned')
    corr, p = similarity(confarray, comparray, simtype='spearmanr')
    if plot:
        tempdf = pd.DataFrame(data={'behavioral confusions (NDE)': comparray, 'model confusions': confarray})
        ax.set_title('spearmanr=%.3f, p=%.3f' % (corr, p))
        sns.regplot(x='behavioral confusions (NDE)', y='model confusions', data=tempdf, ax=ax)
    return corr, p


def visualizeclassification(result, orderedlabels, figsize=[12, 3], comparisons=None):
    '''plot classification results for a single model'''
    confmat, labelaccs = result.confmat, result.labelaccs
    if comparisons:
        f, ax = plt.subplots(1, 5, figsize=figsize)
    else:
        f, ax = plt.subplots(1, 2, figsize=figsize)
    plotmatrix(confmat, title='confusion matrix', ylabel='label', xlabel='guesses', xticklabels=orderedlabels,
               yticklabels=orderedlabels, ax=ax[0])
    plotbar(labelaccs, xlabel='true label', ylabel='percent correct (%)', title='accuracies', xticklabels=orderedlabels,
            ax=ax[1], ylim=[0, 1])
    if comparisons:
        #compare emoaccs
        axis = ax[2]
        rawvals = [comparisons['csummary'].accuracy.ix[emo] for emo in orderedlabels]
        tempdf = pd.DataFrame(data={'raw accuracy': rawvals, 'model accuracy': [el for el in labelaccs]})
        r, p = scipy.stats.pearsonr(tempdf['raw accuracy'], tempdf['model accuracy'])
        axis.set_title('pearsonr=%.3f, p=%.3f' % (r, p))
        sns.regplot(x='raw accuracy', y='model accuracy', data=tempdf, ax=axis)
        #compare to confusions from NDE
        axis = ax[3]
        compareconfmats(confmat, orderedlabels, comparisons, plot=True, ax=axis)
        axis = ax[4]
        ndeitems, ndeaccs = comparisons['summary']['item'].values, comparisons['summary']['accuracy'].values
        modelitems, modelaccs = result.exemplaraccs['item'].values, result.exemplaraccs['mean_acc'].values
        tempdf = pd.DataFrame(index=ndeitems, data={'raw accuracy': ndeaccs})
        tempdf.ix[modelitems, 'model accuracy'] = modelaccs
        r, p = scipy.stats.pearsonr(tempdf['raw accuracy'], tempdf['model accuracy'])
        axis.set_title('pearsonr=%.3f, p=%.3f' % (r, p))
        sns.regplot(x='raw accuracy', y='model accuracy', data=tempdf, ax=axis)


#rsa stuff

def plotrsadict(rsadict, item2emomapping, title, collapse='emo', order=[], ax=None):
    '''function to plot a feature space'''
    rsamatrix, labels, features = rsadict['itemavgs'], rsadict['itemlabels'], rsadict['features']
    df = pd.DataFrame(data={'item': labels, 'emo': [item2emomapping[l] for l in labels]})
    for fn, f in enumerate(features):
        df[f] = [line[fn] for line in rsamatrix]
    avgs = df.groupby(collapse).mean()[features]
    if len(order) > 0:
        avgs = avgs.loc[order]
    if not ax:
        f, ax = plt.subplots(figsize=[4, 2.5])
    if len(features) == 1:
        avgs.plot(kind='barh', legend=False, ax=ax)
    else:
        plotmatrix(avgs.values, ax=ax)
        ax.set_xticks(np.arange(len(features)) + .5)
        ax.set_yticks(range(len(avgs.index.values)))
        ax.set_xticklabels(features, rotation=90)
        ax.set_yticklabels(avgs.index.values)
    ax.set_title(title)


def visualizeinputs(inputspaces, orderedemos, item2emomapping, ncols=3, collapse='emo'):
    print "************INPUT FEATURE SPACES*************"
    numspaces = len(inputspaces.keys())
    f, ax = plt.subplots(int(np.ceil(numspaces / float(ncols))), ncols, figsize=[16, numspaces])
    for keyn, key in enumerate(inputspaces.keys()):
        axis = ax[keyn / ncols, keyn % ncols]
        plotrsadict(inputspaces[key], item2emomapping, key, collapse='emo', order=orderedemos, ax=axis)
    plt.tight_layout()
    plt.show()
    return ax


def visualizeRDMs(allrsasimspaces, orderedemos, ncols=3, keys=[]):
    print "************REPRESENTATIONAL DISSIMILARITY MATRICES***********"
    if len(keys) == 0:
        keys = allrsasimspaces.keys()
    numspaces = len(keys)
    keys_raw = [k for k in keys if 'rdm' in k]
    keys_conf_None = [k for k in keys if 'None' in k]
    keys_conf_item = [k for k in keys if 'item' in k]
    keys = keys_raw + keys_conf_None + keys_conf_item
    f, ax = plt.subplots(int(np.ceil(numspaces / float(ncols))), ncols, figsize=[16, numspaces])
    for keyn, key in enumerate(keys):
        axis = ax[keyn / ncols, keyn % ncols]
        rsamat, labels = allrsasimspaces[key]['simmat_across'], allrsasimspaces[key]['labels']
        plotmatrix(rsamat, ax=axis, xticklabels=labels, yticklabels=labels, title=key)
    plt.tight_layout()
    plt.show()
    return ax


###################################
#main results plots
###################################

def plotacccomparison(resultsdict, keys=[], flags=[], benchmark=None, chance=None, modelcolors=[]):
    '''plot meanacc for each model space'''
    if len(keys) == 0:
        allkeys = [k[0:k.index('_')] for k in resultsdict.keys()]
    else:
        allkeys = [k[0:k.index('_')] for k in keys]
    keys = []
    for k in allkeys:
        if k not in keys:
            keys.append(k)
    if len(flags) == 0:
        flags = list(set([k[0:k.index('_')] for k in resultsdict.keys()]))
    f, ax = plt.subplots(1, len(flags), sharey=True)
    f.suptitle('model comparisons')
    ax[0].set_ylabel('accuracy (mean % across iterations)')
    for fn, f in enumerate(flags):
        if f:
            xlabel = "generalization across %s" % f
        else:
            xlabel = "no generalization"
        keys = [k for k in keys if 'cosine' not in k]
        accs = [resultsdict[k + '_result_' + f].meanacc for k in keys]
        if modelcolors:
            colors = [modelcolors[k + '_confs_' + f] for k in keys]
        axis = ax[fn]
        if modelcolors:
            axis.bar(range(len(accs)), accs, color=colors)
        else:
            axis.bar(range(len(accs)), accs)
        axis.set_xticks(np.arange(len(keys)) + .5)
        axis.set_xticklabels(keys, rotation=90)
        axis.set_xlabel(xlabel)
        if benchmark:
            axis.plot(axis.get_xlim(), [benchmark, benchmark], linestyle='--', alpha=.4)
        if chance:
            axis.plot(axis.get_xlim(), [chance, chance], linestyle='--', alpha=.4, color='#CC5500')
        sns.despine(ax=axis, top=True, right=True, left=False, bottom=False, trim=False)


def plotcorrelationcomparison(resultsdict, orderedlabels, keys=[], flags=[], figsize=[12, 3], unit='items',
                              comparisons=None, modelcolors=[]):
    '''plot item-wise correlation for each model'''
    if len(keys) == 0:
        allkeys = [k[0:k.index('_')] for k in resultsdict.keys()]
    else:
        allkeys = [k[0:k.index('_')] for k in keys]
    keys = []
    for k in allkeys:
        if k not in keys and 'cosine' not in k:
            keys.append(k)
    if len(flags) == 0:
        flags = list(set([k[0:k.index('_')] for k in resultsdict.keys()]))
    f, ax = plt.subplots(1, len(flags), sharey=True)
    f.suptitle('model comparisons: item-wise correlations')
    ax[0].set_ylabel('item-wise accuracy correlation (mean % across iterations)')
    for fn, f in enumerate(flags):
        if f:
            xlabel = "generalization across %s" % f
        else:
            xlabel = "no generalization"
        results = [resultsdict[k + '_result_' + f] for k in keys]
        if modelcolors:
            colors = [modelcolors[k + '_confs_' + f] for k in keys]
        resultsrs, resultsps, resultsns = [], [], []
        for result in results:
            confmat, labelaccs = result.confmat, result.labelaccs
            if unit == 'emos':
                rawvals = [comparisons['csummary'].accuracy.ix[emo] for emo in orderedlabels]
                tempdf = pd.DataFrame(data={'raw accuracy': rawvals, 'model accuracy': [el for el in labelaccs]})
                n = len(ndeitems)
                r, p, n = nonnancorr(tempdf['raw accuracy'].values, tempdf['model accuracy'].values)
                resultsrs.append(r);
                resultsps.append(p);
                resultsns.append(n)
            elif unit == 'items':
                ndeitems, ndeaccs = comparisons['summary']['item'].values, comparisons['summary']['accuracy'].values
                modelitems, modelaccs = result.exemplaraccs['item'].values, result.exemplaraccs['mean_acc'].values
                tempdf = pd.DataFrame(index=ndeitems, data={'raw accuracy': ndeaccs})
                tempdf.ix[modelitems, 'model accuracy'] = modelaccs
                n = len(ndeitems)
                r, p, n = nonnancorr(tempdf['raw accuracy'].values, tempdf['model accuracy'].values)
                resultsrs.append(r);
                resultsps.append(p);
                resultsns.append(n)
            axis = ax[fn]
        if modelcolors:
            axis.bar(range(len(resultsrs)), resultsrs, color=colors)
        else:
            axis.bar(range(len(resultsrs)), resultsrs)
        axis.set_xticks(np.arange(len(keys)) + .5)
        axis.set_xticklabels(keys, rotation=90)
        axis.set_xlabel(xlabel)
        sns.despine(ax=axis, top=True, right=True, left=False, bottom=False, trim=False)


def plotgroupneuralmodelcorrs(modelcorrs, sems, robj, models, colors, ylim):
    ax = simplebar(modelcorrs, yerr=sems, title='%s-- %s of group data' % (robj.roi, robj.corrtype), xlabel='models',
                   xticklabels=models, ylabel='%s\nSEM from bootstrap test' % (robj.corrtype), colors=colors, ylim=ylim,
                   show=False)
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False, trim=False)
    plt.show()
    return ax


def plotindneuralmodelocrrs(modelcorrs, sems, robj, errtype, models, colors, ylim, noiseceiling, benchmark=None):
    if errtype == 'ws':
        eblabel = '+/- 1 SEM (within subjects)'
    else:
        eblabel = '+/- 1 SEM (between subjects)'
    ax = simplebar(modelcorrs, yerr=sems, title='%s-- avg %s of ind data' % (robj.roi, robj.corrtype), xlabel='models',
                   xticklabels=models, ylabel='%s\n%s' % (robj.corrtype, eblabel), colors=colors, ylim=ylim, show=False)
    if noiseceiling:
        ax.axhspan(noiseceiling[0] - .01, noiseceiling[1], facecolor='#8888AA', alpha=0.15)
    if benchmark:
        ax.plot(ax.get_xlim(), [benchmark, benchmark], linestyle='--', alpha=.4)
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False, trim=False)
    plt.show()
    return ax


def plottcresults(tcobjs, roilist):
    import seaborn as sns

    for roi in tcobjs.keys():
        if roi in roilist:
            tcobj = tcobjs[roi]
            colors = sns.color_palette('PuBuGn', len(tcobj.models))
            allmodelcolors = cfg.vizcfg.modelcolors
            tcobj.colordict = {model: '#BBBBBB' for modeln, model in enumerate(tcobj.models)}
            for key in allmodelcolors.keys():
                if key in cfg.vizcfg.modelkeys:
                    tcobj.colordict[key] = allmodelcolors[key]
            f, ax = plt.subplots()
            for model in tcobj.models:
                if model in cfg.vizcfg.modelkeys:
                    tcobj.plottimecourse(model, ax=ax, plotnc=False, ylim=[-.06, .1])

#############################
#dimensionality reduction
#############################

def plotscree(ca, ax=0):
    if ax == 0:
        f, ax = plt.subplots()
    else:
        ax.plot(ca.explained_variance_ratio_)
        ax.set_title('scree plot')
        ax.set_ylabel('variance explained')
        ax.set_xlabel('eigenvector')


def plot3d(matrix, colors=['b'], dimname='dimension', figsize=[4, 3]):
    fig = plt.figure(figsize=figsize)
    ax = Axes3D(fig, elev=-150, azim=110)
    ax.scatter(matrix[:, 0], matrix[:, 1], matrix[:, 2], c=colors, cmap=plt.cm.spectral)
    if type(dimname) == str:
        dimname = [dimname for el in range(3)]
    ax.set_xlabel('%s #%s' % (dimname[0], 0))
    ax.set_ylabel('%s #%s' % (dimname[1], 1))
    ax.set_zlabel('%s #%s' % (dimname[2], 2))
    return ax


def plot2d(matrix, colors=['b'], dimname='dimension', figsize=[4, 3]):
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(matrix[:, 0], matrix[:, 1], c=colors, cmap=plt.cm.spectral)
    if type(dimname) == str:
        dimname = [dimname for el in range(2)]
    ax.set_xlabel('%s #%s' % (dimname[0], 0))
    ax.set_ylabel('%s #%s' % (dimname[1], 1))
    return ax


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            FancyArrowPatch.draw(self, renderer)


def plotcomponentcorrs(obsscores, ax=None, label=''):
    #sns.corrplot(pd.DataFrame(obsscores), method='pearson', annot=False, ax=ax)
    correlations = pd.DataFrame(obsscores).corr(method='pearson').values
    #plotmatrix(correlations)
    corrs = correlations[np.tril_indices(len(correlations), 1)]
    plotmatrix(np.tril(correlations, -1), colorbar=False, ax=ax, xlabel=label)
    ax.set_title('correlation of components in raw observation space\n mean corr= %.6f' % (np.mean(corrs)))
    plt.tight_layout()


def plotcompsininputspace(matrix, axislabels, dimloadings, varexplained, mapping=None, dimindices=[0, 1, 2],
                          dimlabels=[],
                          componentindices=[0, 1, 2], title=None, constant=None):  #plot observations on 3 components
    #plot 3 components in input dimension space
    data = matrix[:, dimindices]
    components = dimloadings.T[:, dimindices]
    if mapping:
        axisdims = list(set(mapping.values()))
        colornums = sns.color_palette('husl', len(axisdims))
        labelcolors = {el: colornums[eln] for eln, el in enumerate(axisdims)}
        colors = [labelcolors[mapping[el]] for el in axislabels]
    else:
        colors = 'b'
    if len(dimlabels) > 0:
        dimname = [dimlabels[el] for el in dimindices]
    else:
        dimname = 'input dim'
    if len(dimindices) == 3:
        ax = plot3d(data, colors=colors, dimname=dimname)
    else:
        ax = plot2d(data, colors=colors, dimname=dimname)
    ar = [np.max(data, axis=0)[index] - np.min(data, axis=0)[index] for index in range(len(dimindices))]
    datamean = np.mean(data, axis=0)
    for x in componentindices:
        drawvector(components, x, datamean, ax, ar, constant)
    plt.show()


def drawvector(eigenvectors, index, datamean, ax, ar, constant):
    eigenvector = eigenvectors[index]
    if not constant:
        constant = 5
    scaler = constant / np.mean(ar)
    x = (datamean[0], (eigenvector[0] * scaler + datamean[0]))
    y = (datamean[1], (eigenvector[1] * scaler + datamean[1]))
    if len(datamean) == 2:
        ax.text(x[1], y[1], index, bbox=dict(boxstyle="round", fc="w"))
        arrow = ax.annotate('', xy=[x[1], y[1]], xycoords='data', xytext=[x[0], y[0]], textcoords='data',
                            arrowprops=dict(facecolor='red', width=1))
    else:
        z = (datamean[2], eigenvector[2] * scaler + datamean[2])
        a = Arrow3D(x, y, z, mutation_scale=10, lw=1, arrowstyle="-|>", color="r")
        ax.add_artist(a)


def plotinputincomponentspace(obsscores, axislabels, mapping=None, componentindices=[0, 1, 2], title=None):
    #plot observations on 3 components
    componentindices = componentindices[:len(obsscores[0])]
    data = obsscores[:, componentindices]
    if mapping:
        axisdims = list(set(mapping.values()))
        colornums = sns.color_palette('husl', len(axisdims))
        labelcolors = {el: colornums[eln] for eln, el in enumerate(axisdims)}
        colors = [labelcolors[mapping[el]] for el in axislabels]
    else:
        colors = 'b'
    if len(obsscores[0]) == 3:
        ax = plot3d(data, colors=colors, dimname='component')
    else:
        ax = plot2d(data, colors=colors, dimname='component')
    if mapping:
        for label, name in mapping.items():
            if len(obsscores[0]) == 3:
                ax.text3D(obsscores[axislabels == label, 0].mean(),
                          obsscores[axislabels == label, 1].mean(),
                          obsscores[axislabels == label, 2].mean(), name,
                          bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
            else:
                ax.text(obsscores[axislabels == label, 0].mean(),
                        obsscores[axislabels == label, 1].mean(), name,
                        bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))


def plotdimensionsincomponentspace(dimloadings, dimlabels, mapping=None, componentindices=[0, 1, 2], title=None):
    #plot dimensions on 3 components
    dimloadings = dimloadings.T
    componentindices = componentindices[:len(dimloadings[0])]
    data = dimloadings[:, componentindices]
    if len(componentindices) == 3:
        ax = plot3d(data, dimname='component')
    else:
        ax = plot2d(data, dimname='component')
    for labeln, label in enumerate(dimlabels):
        if len(componentindices) == 3:
            ax.text3D(dimloadings[labeln][0], dimloadings[labeln][1], dimloadings[labeln][2], label,
                      bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
        else:
            ax.text(dimloadings[labeln][0], dimloadings[labeln][1], label,
                    bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))