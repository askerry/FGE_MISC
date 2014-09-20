# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 17:23:56 2014

@author: amyskerry
"""
import os

from sklearn import svm, cluster
from sklearn.cross_decomposition import PLSRegression
from sklearn.cluster import AffinityPropagation


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
        model = svm.SVC(C=.01, kernel='linear')  #svm.SVC implements multiclass as one vs. one
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

vizcfg = Config()

vizcfg.colordict = {'benchmark': '#3344AA', 'dimmodels': '#5577CC', 'control': '#228855', 'textsentiment': '#00C444',
                    'other': '#BBCCCC'}
vizcfg.vizmodels = ['NDEconfmat_raw',
                    'NDEconfmat_rdm',
                    'NDEconfmat_confs_item',
                    'NDEconfmat_confs_None',
                    'explicits_rdm',
                    'explicits_confs_item',
                    'explicits_confs_None',
                    '37dim_rdm',
                    '37dim_confs_item',
                    '37dim_confs_None',
                    'basicemo_rdm',
                    'basicemo_confs_item',
                    'basicemo_confs_None',
                    'valencearousal_rdm',
                    'valencearousal_confs_item',
                    'valencearousal_confs_None',
                    'arousal_rdm',
                    'arousal_confs_item',
                    'arousal_confs_None',
                    'valence_rdm',
                    'valence_confs_item',
                    'valence_confs_None',
                    'naivebayes_rdm',
                    'naivebayes_confs_item',
                    'naivebayes_confs_None',
                    'dlsentiment_rdm',
                    'dlsentiment_confs_item',
                    'dlsentiment_confs_None',
                    'cohmetrix_rdm',
                    'cohmetrix_confs_item',
                    'cohmetrix_confs_None',
                    'cosinesim_rdm',
                    'intensity_rdm',
                    'intensity_confs_item',
                    'intensity_confs_None']
vizcfg.excludemodels = ['NDEconfmat_rdm', 'explicits_rdm']
vizcfg.flags=['rdm']
vizcfg.modelkeys = [m for m in vizcfg.vizmodels if m not in vizcfg.excludemodels]
for f in vizcfg.flags:
    vizcfg.modelkeys= [m for m in vizcfg.modelkeys if f in m]
vizcfg.allmodeltypes = {'37dim_rdm': 'dimmodels',
                        'arousal_rdm': 'dimmodels',
                        'valence_rdm': 'dimmodels',
                        'cohmetrix_rdm': 'control',
                        '37dim_confs_item': 'dimmodels',
                        '37dim_confs_None': 'dimmodels',
                        'naivebayes_confs_item': 'textsentiment',
                        'naivebayes_confs_None': 'textsentiment',
                        'explicits_confs_item': 'benchmark',
                        'explicits_confs_None': 'benchmark',
                        'basicemo_rdm': 'dimmodels',
                        'basicemo_confs_item': 'dimmodels',
                        'basicemo_confs_None': 'dimmodels',
                        'explicits_rdm': 'benchmark',
                        'cohmetrix_confs_item': 'control',
                        'cohmetrix_confs_None': 'control',
                        'dlsentiment_rdm': 'textsentiment',
                        'valence_confs_item': 'dimmodels',
                        'valence_confs_None': 'dimmodels',
                        'cosinesim_rdm': 'control',
                        'intensity_rdm': 'other',
                        'naivebayes_rdm': 'textsentiment',
                        'NDEconfmat_rdm': 'benchmark',
                        'NDEconfmat_raw': 'benchmark',
                        'NDEconfmat_confs_None': 'benchmark',
                        'NDEconfmat_confs_item': 'benchmark',
                        'arousal_confs_item': 'dimmodels',
                        'intensity_confs_item': 'other',
                        'dlsentiment_confs_item': 'textsentiment',
                        'valencearousal_confs_item': 'dimmodels',
                        'arousal_confs_None': 'dimmodels',
                        'intensity_confs_None': 'other',
                        'dlsentiment_confs_None': 'textsentiment',
                        'valencearousal_confs_None': 'dimmodels',
                        'valencearousal_rdm': 'dimmodels'}
vizcfg.modelcolors = {key: vizcfg.colordict[vizcfg.allmodeltypes[key]] for key in vizcfg.allmodeltypes.keys()}

cohmetrixcfg = Config()
cohmetrixcfg.excludecols = ['qnum_str', 'cond', 'stimname', 'text', 'qnum',
                            "DESPC 'Paragraph count, number of paragraphs'",
                            "DESPL 'Paragraph length, number of sentences in a paragraph, mean'",
                            "DESPLd 'Paragraph length, number of sentences in a pragraph, standard deviation'"]

cohmetrixcfg.cols = ["RDFRE 'Flesch Reading Ease'",
                     "PCREFz 'Text Easability PC Referential cohesion, z score'",
                     #"SMCAUSv 'Causal verb incidence'",
                     # "SMCAUSr 'Ratio of casual particles to causal verbs'",
                     #"PCCNCz 'Text Easability PC Word concreteness, z score'",
                     "DRNEG 'Negation density, incidence'",
                     "SYNNP 'Number of modifiers per noun phrase, mean'",
                     "SYNLE 'Left embeddedness, words before main verb, mean'"]


tickcfg={}
tickcfg['emos']=cfg.condmapping
tickcfg['models']={}#