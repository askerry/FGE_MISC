# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 08:18:21 2014

@author: amyskerry
"""
import sys
sys.path.append('/mindhive/saxelab/scripts/aesscripts/')
import pandas as pd
import numpy as np
import os
import FGE_MISC.code.behave.NDIM as ndimf
from copy import deepcopy
import FGE_MISC.code.vizfuncs as viz
from FGE_MISC.code.config import rootdir
import string
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews, stopwords
from nltk.stem.porter import PorterStemmer

# ############################################################################
#general
#############################################################################

def makedataframe(itemavgs, itemlabels, item2emomapping):
    data = {'item': itemlabels, 'emo': [item2emomapping[item] for item in itemlabels]}
    nfeatures = len(itemavgs[0])
    for featuren in range(nfeatures):
        data[featuren] = [line[featuren] for line in itemavgs]
    return pd.DataFrame(data=data)


#############################################################################
#Feature spaces
#############################################################################

#simplest bagofwords model
def bagofwords(df, item2emomapping):
    '''creates simple bag of words feature space'''
    print "creating bag-of-words feature space"
    from sklearn.feature_extraction.text import CountVectorizer

    listofstrings = list(df['cause'].values)
    itemlabels = ['q%0.f' % qnum for qnum in df['qnum'].values]
    vectorizer = CountVectorizer(min_df=1)
    analyzer = vectorizer.build_analyzer()
    bagofwords = vectorizer.fit_transform(listofstrings)
    features = vectorizer.get_feature_names()
    bagofwords = bagofwords.toarray()
    itemavgs = [list(line) for line in bagofwords]
    ndf = makedataframe(itemavgs, itemlabels, item2emomapping)
    return ndf


#smarter tf-idf based text similarity
def tfidf(df, item2emomapping, visualize=False, computecosine=False):
    print "creating tf-idf feature space"
    import nltk
    from nltk.corpus import stopwords
    import string
    from nltk.stem.porter import PorterStemmer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel

    def stem_tokens(tokens, stemmer):
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return stemmed

    def tokenize(text, stemmer=None):
        #stemmer=PorterStemmer()
        tokens = nltk.word_tokenize(text)
        filtered = [w for w in tokens if not w in stopwords.words('english')]
        if stemmer:
            stems = stem_tokens(filtered, stemmer)
        else:
            stems=filtered
        return stems

    def singletextsimilarity(tfidf, index, listofstrings, printdeets=False):
        cosine_similarities = linear_kernel(tfidf[index], tfidf).flatten()
        if printdeets:
            most_related_docs_indices = cosine_similarities.argsort()[:-5:-1]
            most_related_similarities = cosine_similarities[most_related_docs_indices]
            print "docs most related to doc #%s are %s." % (
            index, ', '.join([str(el) for el in most_related_docs_indices]))
            print "there similarities are %s." % (', '.join([str(el) for el in most_related_similarities]))
        return cosine_similarities

    def nandiagmatrix(matrix):
        nanmatrix = np.array(deepcopy(matrix))
        nanmatrix[np.diag_indices(len(nanmatrix))] = np.nan
        return list(nanmatrix)

    def FGEtfidf(df, computecosine=False):
        tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
        tfs = tfidf.fit_transform(listofstrings)
        simmatrix = []
        if computecosine:
            for ln, l in enumerate(listofstrings):
                similarities = singletextsimilarity(tfs, ln, listofstrings)
                simmatrix.append(similarities)
                simmatrix = nandiagmatrix(simmatrix)
        return simmatrix, tfs

    listofstrings = list(df['cause'].values)
    itemlabels = ['q%0.f' % qnum for qnum in df['qnum'].values]
    simmatrix, tfs = FGEtfidf(listofstrings, computecosine)
    simoutput = {'matrix': simmatrix, 'itemlabels': itemlabels}
    if visualize and computecosine:
        ndimf.plotmatrix(np.array(simmatrix), xlabel='diag non-independent')
    dense=np.array([np.array(line.todense()).flatten() for line in tfs])
    df = makedataframe(dense, itemlabels, item2emomapping)
    ndimf.quicksave(df, os.path.join(rootdir,'data/stimdfs','tfidfdf.pkl'))
    return simoutput, df


#naive bayes classifier on bag of words, trained on rotten tomatoes dataset
def sentimentbow(df, item2emomapping, trainset=[], stemmer=PorterStemmer()):
    print "creating valence feature space based on naive bayes classifier trained on rotten tomatoes dataset"
    def stem_tokens(tokens, stemmer):
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return stemmed

    def preptokens(tokens, stemmer):
        filtered = [w for w in tokens if not w in stopwords.words('english') and not w in string.punctuation]
        if stemmer:
            stems = stem_tokens(filtered, stemmer)
        else:
            stems=filtered
        finaltokens = {word: True for word in stems}
        return finaltokens

    def bagofwordsclassifier(tokenizer, trainfeats, teststrings, stemmer):
        testtokens = [string.split(' ') for string in teststrings]
        testfeats = [tokenizer(tokens, stemmer) for tokens in testtokens]
        print "training NB classifier on %s item training set." % (len(trainfeats))
        print "testing on %s items (FGE)" % (len(testfeats))
        classifier = NaiveBayesClassifier.train(trainfeats)
        predictions, probabilities = [], []  #of pos, neg
        for story in testfeats:
            label = classifier.classify(story)
            pdist = classifier.prob_classify(story)
            probabilities.append((pdist.prob('pos'), pdist.prob('neg')))
            predictions.append(label)
        return predictions, probabilities

    def maketrainset(movie_reviews, tokenizer, stemmer):
        negids = movie_reviews.fileids('neg')
        posids = movie_reviews.fileids('pos')
        negfeats = [(tokenizer(movie_reviews.words(fileids=[f]), stemmer), 'neg') for f in negids]
        posfeats = [(tokenizer(movie_reviews.words(fileids=[f]), stemmer), 'pos') for f in posids]
        trainfeats = negfeats + posfeats
        return trainfeats

    ##run it
    listofstrings = list(df['cause'].values)
    itemlabels = ['q%0.f' % qnum for qnum in df['qnum'].values]
    if len(trainset) == 0:
        trainset = maketrainset(movie_reviews, preptokens, stemmer)
    predictions, probabilities = bagofwordsclassifier(preptokens, trainset, listofstrings, stemmer)
    itemavgs = [[p[0]] for p in probabilities]
    ndf = makedataframe(itemavgs, itemlabels, item2emomapping)
    ndimf.quicksave(ndf, os.path.join(rootdir,'data/stimdfs','sentbowdf.pkl'))
    return ndf

#similarity in NDE confusions:
def computeNDEspace(NDE_output, item2emomapping):
    itemavgs=NDE_output['propconfusions'][[col for col in NDE_output['propconfusions'].columns if col !='Neutral']]
    itemavgs=itemavgs.values
    itemlabels=list(NDE_output['propconfusions'].index.values)
    ndf = makedataframe(itemavgs, itemlabels, item2emomapping)
    ndimf.quicksave(ndf, os.path.join(rootdir,'data/stimdfs','NDEconfdf.pkl'))
    return ndf

#cohmetrix
def computecohmetrixspace(df, item2emomapping, cols=[], excludecols=[]):
    print "creating cohmetrix feature space"
    if len(cols) == 0:
        cols = [col for col in df.columns]
    cols = [col for col in cols if col not in excludecols]
    itemlabels = ['q%.0f' % qnum for qnum in df.qnum[0:200].values]
    df = df[[col for col in cols if df[col].dtype == 'float64']]
    nonnanstds = [col for col in df.columns if df[col].std() > 0]
    df = df[nonnanstds]
    df = (df - df.mean()) / (df.std())  #zscore each column
    itemavgs = [line for line in df.values][0:200]
    ndf = makedataframe(itemavgs, itemlabels, item2emomapping)
    ndimf.quicksave(ndf, os.path.join(rootdir,'data/stimdfs','cohmetrixdf.pkl'))
    return ndf


#behavioral intensity
def behavioralintensity(df, item2emomapping):
    print "creating intensity feature space (based on in scanner behavioral responses)"
    grouped = df.groupby('qnum').mean()
    itemlabels = list(grouped.index.values)
    itemavgs = [[el] for el in list(grouped['intensities'].values)]
    ndf = makedataframe(itemavgs, itemlabels, item2emomapping)
    ndimf.quicksave(ndf, os.path.join(rootdir,'data/stimdfs','intensitydf.pkl'))
    return ndf


def sentimentrntn(df, item2emomapping):
    print "creating valence feature space based on Socher et al's sentiment analysis project"
    df = df[df['qnum'] <= 200]
    itemlabels = list(['q%.0f' % qnum for qnum in df['qnum'].values])
    itemavgs = [[el] for el in list(df['stimsavg'].values)]
    ndf = makedataframe(itemavgs, itemlabels, item2emomapping)
    ndimf.quicksave(ndf, os.path.join(rootdir,'data/stimdfs','sentrntndf.pkl'))
    return ndf