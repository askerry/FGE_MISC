# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 17:23:56 2014

@author: amyskerry
"""
import os

from sklearn import svm, cluster
from sklearn.cross_decomposition import PLSRegression
from sklearn.cluster import AffinityPropagation
from sklearn.decomposition import PCA, FastICA
import numpy as np
from collections import OrderedDict as odict


class DataFiles():
    def __init__(self, rootdir):
        self.rootdir = rootdir

    def printfiles(self):
        attributes = [el for el in dir(self) if el not in dir(Config) and el != 'printfiles']
        for a in attributes:
            print "%s: %s" % (a, getattr(self, a))


class Config():
    def printdeets(self):
        attributes = [el for el in dir(self) if el not in dir(Config) and el != 'printdeets']
        for a in attributes:
            vals = getattr(self, a)
            if type(vals) in (str, int, float, bool):
                print "%s: %s" % (a, vals)
            elif type(vals) == list:
                print "%s: %s" % (a, ', '.join(vals))
            elif type(vals) == dict:
                print "%s: %s" % (a, ', '.join(str(el) for el in vals.items()))
            else:
                print "%s:" % a
                print vals
        pass


# configure main modeling parameters here
class ModelConfig():
    def SVM_model(self):
        model = svm.SVC(C=1, kernel='linear')  #svm.SVC implements multiclass as one vs. one
        #SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,gamma=0.0, kernel='rbf', max_iter=-1, probability=False, random_state=None,shrinking=True, tol=0.001, verbose=False)
        #model=svm.LinearSVC() #svm.LinearSVC() implements multiclass as one vs. test
        #see http://scikit-learn.org/stable/modules/svm.html
        return model

    def PLSR_model(self, ncomp):
        #PLS regression is particularly suited when the matrix of predictors has more variables than observations, and when there is multicollinearity among X values. By contrast, standard regression will fail in these cases.
        model = PLSRegression(self, n_components=ncomp, scale=True)
        return model

    def KMEANS_model(self, numclust):
        model = cluster.KMeans(n_clusters=numclust)
        return model

    def affinityprop(self):
        ap = AffinityPropagation(damping=0.8, max_iter=500, affinity='euclidean', convergence_iter=25, copy=True,
                                 preference=None)
        return ap

    def PCA(self):
        pca = PCA()
        return pca

    def ICA(self):
        ica = FastICA()
        return ica

    def printmodelparams(self):
        print "not implemented yet"

#set up config info to import into analysis
mcfg = ModelConfig()

#rootdir = '/Users/amyskerry/dropbox/FGE_MISC/'
rootdir = '/mindhive/saxelab/scripts/aesscripts/FGE_MISC/'
dfiles = DataFiles(rootdir)

dfiles.ndefile = os.path.join(rootdir, 'data',
                              'NDE.csv')  #contains NDEdl.csv and the first row of the two woops (with checks manually corrected since these subjects didn't have Neutral option)
dfiles.ndimfile = os.path.join(rootdir, 'data', 'NDIM.csv')
dfiles.ndimcfile = os.path.join(rootdir, 'data', 'NDIMc.csv')
dfiles.ndimasdfile = os.path.join(rootdir, 'data', 'NDIMASD.csv')
dfiles.stimfile = os.path.join(rootdir, 'data', 'NDE_stims.csv')
dfiles.appraisalfile = os.path.join(rootdir, 'data', 'appraisals.csv')
dfiles.appraisalcfile = os.path.join(rootdir, 'data', 'appraisalsC.csv')
dfiles.sentimentfile = os.path.join(rootdir, 'data', 'stimdata', 'FGE_stims_sentiment.csv')
dfiles.cohmetrixfile = os.path.join(rootdir, 'data', 'stimdata', 'FGE_stims_cohmetrix.csv')
dfiles.FGEstimfile = os.path.join(rootdir, 'data', 'stimdata', 'FGE_stims.csv')
dfiles.intensityfile = os.path.join(rootdir, 'data', 'stimdata', 'behavioralintensity.csv')

############################# general analysis configuration details ############################

cfg = Config()
cfg.allorderedemos = ['Grateful', 'Joyful', 'Hopeful', 'Excited', 'Proud', 'Impressed', 'Content', 'Nostalgic',
                      'Surprised', 'Lonely', 'Furious', 'Terrified', 'Apprehensive', 'Annoyed', 'Guilty', 'Disgusted',
                      'Embarrassed', 'Devastated', 'Disappointed', 'Jealous', 'Neutral']
cfg.orderedemos = [e for e in cfg.allorderedemos if e != 'Neutral']
cfg.condmapping = {'Devastated': 'DEV', 'Disappointed': 'DIP', 'Annoyed': 'ANN', 'Apprehensive': 'APP',
                   'Disgusted': 'DIG', 'Embarrassed': 'EMB', 'Furious': 'FUR', 'Guilty': 'GUI', 'Jealous': 'JEA',
                   'Terrified': 'TER', 'Lonely': 'LON', 'Excited': 'EXC', 'Joyful': 'JOY', 'Proud': 'PRO',
                   'Content': 'CON', 'Grateful': 'GRA', 'Hopeful': 'HOP', 'Impressed': 'IMP', 'Nostalgic': 'NOS',
                   'Surprised': 'SUR'}
cfg.subjcols = ['city', 'country', 'age', 'gender', 'thoughts', 'response_noface', 'response_intune',
                'response_nothought', 'response_needverbal', 'response_facevoice', 'response_surprised']
cfg.fixedcols = ['item', 'emo', 'subjid', 'rownum']
cfg.appraisalsubsets = {'be': ['afraid_dim', 'happy_dim', 'angry_dim', 'sad_dim', 'surprised_dim', 'disgusted_dim'],
                        'va': ['valence', 'arousal'], 'v': ['valence'], 'a': ['arousal']}
cfg.ndechecks = {'q201': 'Neutral', 'q202': 'Neutral'}
cfg.ndimchecks = {'main_character': [7, 5]}  #avgthresh, indthresh
cfg.asdchecks = {'main_character': [0, 0]}  #avgthresh, indthresh (different exclusion criteria)
cfg.ndevisualize = False
cfg.ndimvisualize = True
cfg.valencedict = {'pos': ['Grateful', 'Joyful', 'Hopeful', 'Excited', 'Proud', 'Impressed', 'Content', 'Nostalgic'],
                   'neg': ['Lonely', 'Furious', 'Terrified', 'Apprehensive', 'Annoyed', 'Guilty', 'Disgusted',
                           'Embarrassed', 'Devastated', 'Disappointed', 'Jealous']}

###################### visualization configurations ########################

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(float(int(value[i:i + lv // 3], 16)) / 255 for i in range(0, lv, lv // 3))
def offset(rgb, offset):
    return tuple([el * offset for el in rgb])

vizcfg = Config()
vizcfg.basecolor = '#4c72b0'
vizcfg.cmap = 'Greys'

colordict = {'benchmark': '#3344AA', 'dimmodels': '#5577CC', 'control': '#228855', 'textsentiment': '#00FF44',
             'other': '#BBCCCC', 'dimreduction': '#CC7788', 'appraisalfeature':'#BBCC88'}
vizcfg.colordict = {item[0]: hex_to_rgb(item[1]) for item in colordict.items()}
models = odict(
    [('behaviorialconfs', 'benchmark'), ('explicits', 'benchmark'), ('37appraisals', 'dimmodels'), ('basicemo', 'dimmodels'),
     ('valencearousal', 'dimmodels'), ('valence', 'dimmodels'), ('arousal', 'dimmodels'),
     ('sentimentBoW', 'textsentiment'), ('sentimentRNTN', 'textsentiment'), ('cohmetrix', 'control'),
     ('syntax', 'control'), ('readingease', 'control'), ('tfidf', 'control'), ('bagofwordsdf', 'control'),
     ('intensity', 'control'), ('1stPC', 'dimreduction'), ('2ndPC', 'dimreduction'), ('3rdPC', 'dimreduction')])
for f in ['dangerfeat','occurredfeat','expectednessfeat','peoplefeat','closeothersfeat','relationshipinfluencefeat','knowledgechangefeat','pressurefeat','selfinvolvementfeat','moralfeat','agentintentionfeat','suddennessfeat','futurefeat','othersknowledgefeat','controlfeat','agentsituationfeat','safetyfeat','selfcausefeat','certaintyfeat','relevancefeat','consequencesfeat','selfesteemfeat','pastfeat','rememberfeat','familiarityfeat','pleasantnessfeat','selfconsistencyfeat','fairnessfeat','goalconsistencyfeat','mentalstatesfeat','agentcausefeat','attentionfeat','alteringfeat','repetitionfeat','copingfeat','freedomfeat','bodilydiseasefeat','psychologicalchangefeat']:
    models[f]='appraisalfeature'
dimmodels=['37appraisals', 'basicemo', 'valencearousal', 'valence', 'arousal']
excludemodels = ['behaviorialconfs', 'explicits', 'cohmetrix', 'bagofwordsdf']
flags = ['rdm', 'confs_item', 'confs_None']
vizcfg.allmodels = odict()
vizcfg.excludemodels,vizcfg.dimmodels = [],[]
for flag in flags:
    for m in excludemodels:
        vizcfg.excludemodels.append(m + '_' + flag)
    for m in models.keys():
        vizcfg.allmodels[m + '_' + flag] = models[m]
    for m in dimmodels:
        vizcfg.dimmodels.append(m + '_' + flag)

vizcfg.vizflag = 'rdm'
vizcfg.modelkeys = [m for m in vizcfg.allmodels.keys() if m not in vizcfg.excludemodels and vizcfg.vizflag in m]
vizcfg.PCAkeys = [m for m in vizcfg.allmodels.keys() if 'PC' in m and vizcfg.vizflag in m]
vizcfg.featkeys = [m for m in vizcfg.allmodels.keys() if 'feat' in m and vizcfg.vizflag in m]

vizcfg.modelcolors = {key: vizcfg.colordict[vizcfg.allmodels[key]] for key in vizcfg.allmodels.keys()}
offsets = [0.85, 0.6, 0.75, 0.5, 0.8, 0.7, 0.9, 0.95, 0.55, 0.65, 0.6, 0.5, 0.75, 0.85, 0.8, 0.7, 0.9, 0.95, 0.55, 0.65, 0.6, 0.5, 0.75, 0.85, 0.8, 0.7, 0.9, 0.95, 0.55, 0.65, 0.6, 0.5, 0.75, 0.85, 0.8, 0.7,0.9, 0.95, 0.55,0.65]
for key in vizcfg.modelkeys:
    vizcfg.modelcolors[key] = offset(vizcfg.colordict[vizcfg.allmodels[key]], offsets.pop())

tickcfg = {}
tickcfg['emos'] = cfg.condmapping
tickcfg['models'] = {}

############ cohmetrix ###############

cohmetrixcfg = Config()
cohmetrixcfg.excludecols = []
cohmetrixcfg.cols = ["RDFRE 'Flesch Reading Ease'", #easability
                     "PCREFz 'Text Easability PC Referential cohesion, z score'", #easability
                     "DRNEG 'Negation density, incidence'", #syntactic
                     "SYNNP 'Number of modifiers per noun phrase, mean'", #syntactic
                     "SYNLE 'Left embeddedness, words before main verb, mean'"]#syntactic
cohmetrixcfg.syntaxcols = ["DRNEG 'Negation density, incidence'",
                           "SYNNP 'Number of modifiers per noun phrase, mean'",
                           "SYNLE 'Left embeddedness, words before main verb, mean'"]
cohmetrixcfg.easecols = ["RDFRE 'Flesch Reading Ease'",
                         "PCREFz 'Text Easability PC Referential cohesion, z score'"]

