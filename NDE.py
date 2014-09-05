# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 11:33:03 2014

@author: amyskerry
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle

#basic prep functions
def setup(ndefile, checks, deletecols):    
    ndedf=pd.read_csv(ndefile)
    print "starting with %s rows" %(len(ndedf))
    ndedf=ndedf[ndedf.submission_date.notnull()]
    print "reduced to %s intact rows" %(len(ndedf))
    for c in checks:
        ndedf=ndedf[ndedf[c].apply(passed, expectedans=checks[c])]
    print "reduced to %s check-passing rows" %(len(ndedf))
    ndedf=deletecols(ndedf, ['q201', 'q202', 'correctA201', 'correctA202', 'otherword'])
    return ndedf
    
def passed(answer, expectedans=None):
    return answer==expectedans
    
def deletecols(df, dlist):
    cols=[c for c in df.columns if not all(df[c].isnull()) and not any([(string in c) for string in dlist])]
    print "reduced from %s to %s columns" %(len(df.columns), len(cols)) 
    return df[cols]
    
#general
def heatmapdf(df, xlabel=None, ylabel=None, title=None, xticklabels=None, yticklabels=None, ax=None):
    if ax:
        pass
    else:
        f,ax=plt.subplots()
    ax.pcolor(df)
    if xticklabels:
        ax.set_xticks(np.arange(len(df.columns))+.5)
        ax.set_xticklabels(df.columns, rotation=90)
        ax.set_xlim([0,len(df.columns)])
    if yticklabels:
        ax.set_yticks(np.arange(len(df.index)))
        ax.set_yticklabels(df.index)
        ax.set_ylim([0,len(df.index)])
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    plt.colorbar(plt.pcolor(df), ax=ax)
    
def reorder(items, emos, orderedemos):
    tups=zip(emos,items)
    rtups=[]
    for e in orderedemos:
        reltups=[el for el in tups if el[0]==e]
        rtups.extend(reltups)
    remos,ritems=zip(*rtups)
    return ritems, remos
    
# make item confusions/summaries
def defineanswerkey(df, orderedemos):
    items=[col for col in df.columns if col[0]=='q']
    answers=[col for col in df.columns if col[:8]=='correctA']
    correctcols=[[a for a in answers if int(i[1:])==int(a[8:])][0] for i in items]
    emos=[df[col].dropna().values[0] for col in correctcols]
    items, emos=reorder(items, emos, orderedemos)
    return items, emos
    
def confusions(df, items, orderedemos):
    rawconfusions=pd.DataFrame(index=items, columns=orderedemos).fillna(0)
    for item in items:
        tempdf=df[item][df[item].notnull()]
        for response in tempdf:
            rawconfusions.loc[item,response]+=1
    propconfusions=rawconfusions.div(rawconfusions.sum(axis=1), axis=0) #normalizes by row (each element/sum of row)
    return rawconfusions, propconfusions
    
def summarize(items, emos, rawconfusions, propconfusions):
    summary=pd.DataFrame(index=range(len(items)))
    summary['item']=items
    summary['answer']=emos
    summary['count']=rawconfusions.sum(axis=1)
    summary['accuracy']= [propconfusions.loc[row['item'],row['answer']] for index,row in summary.iterrows()]
    print "overall accuracy: %.3f%%" %(summary['accuracy'].mean())
    return summary

def score(df, orderedemos):
    items,emos=defineanswerkey(df, orderedemos)
    rawconfusions, propconfusions=confusions(df, items, orderedemos)
    summary=summarize(items, emos, rawconfusions, propconfusions)
    return rawconfusions,propconfusions,summary

#collapse functions
def collapseconf(confusions, summary, orderedemos):
    items, emos=summary['item'], summary['answer']
    tups=zip(emos,items)
    collapsed=pd.DataFrame(index=confusions.columns, columns=confusions.columns).fillna(0)
    for e in orderedemos:
        relemos, relitems=zip(*[el for el in tups if el[0]==e])
        tempdf=confusions.loc[relitems,:]
        collapsed.loc[e,:]=tempdf.mean(axis=0)
    return collapsed
    
def collapsesummary(summary, orderedemos):
    items, emos=summary['item'], summary['answer']
    tups=zip(emos,items)
    collapsed=pd.DataFrame(index=orderedemos, columns=summary.columns).fillna(0)
    for e in orderedemos:
        relemos, relitems=zip(*[el for el in tups if el[0]==e])
        tempdf=summary[[item in relitems for item in summary['item']]]
        collapsed.loc[e,:]=tempdf.mean(axis=0)
        collapsed.loc[e,'SEM']=tempdf['accuracy'].std(axis=0)/np.sqrt(len(tempdf['accuracy'].dropna()))
    return collapsed
    
#main
def main(ndefile, orderedemos, checks, visualize=True):
    print "running setup for NDE..."
    ndedf=setup(ndefile, checks, deletecols)
    print
    print "item-wise results:"
    rawconfusions,propconfusions,summary=score(ndedf, orderedemos)
    if visualize:
        f,ax=plt.subplots(1,2, figsize=[10,4])
        heatmapdf(rawconfusions, xlabel='guessed emotion', ylabel='item', title='raw confusions', xticklabels=orderedemos, ax=ax[0])
        heatmapdf(propconfusions, xlabel='guessed emotion', ylabel='item', title='proportions',xticklabels=orderedemos, ax=ax[1])
        f=plt.figure(figsize=[16,2])
        summary['accuracy'].plot(kind='bar')
        plt.show()
    orderedemos=[el for el in orderedemos if el !='Neutral'] #we don't want neutral anymore
    print
    print "collapsing over items:"
    craw=collapseconf(rawconfusions, summary, orderedemos)
    cprop=collapseconf(propconfusions, summary, orderedemos)
    csummary=collapsesummary(summary, orderedemos)
    if visualize:
        f,ax=plt.subplots(1,3, figsize=[18,4])
        heatmapdf(craw, xlabel='guessed emotion', ylabel='item', title='raw confusions', xticklabels=orderedemos, yticklabels=orderedemos, ax=ax[0])
        heatmapdf(cprop, xlabel='guessed emotion', ylabel='item', title='proportions',xticklabels=orderedemos, yticklabels=orderedemos, ax=ax[1])
        ax[2].clear()
        csummary['accuracy'].plot(kind='bar', yerr=csummary['SEM'], ax=ax[2])
        ax[2].set_ylabel('accuracy(+/- SEM across items)')
    NDE_output={'rawconfusions':rawconfusions,'propconfusions':propconfusions,'summary':summary,'craw':craw,'cprop':cprop,'csummary':csummary}
    print "NDE analysis complete."    
    return NDE_output, orderedemos
    
if __name__=='__main__':
    #fix this hard coding
    rootdir='/Users/amyskerry/dropbox/FGE_MISC/'
    ndefile=os.path.join(rootdir, 'data', 'NDE.csv') #contains NDEdl.csv and the first row of the two woops (with checks manually corrected since these subjects didn't have Neutral option)
    orderedemos=['Grateful', 'Joyful','Hopeful','Excited','Proud','Impressed','Content','Nostalgic', 'Surprised','Lonely', 'Furious','Terrified','Apprehensive','Annoyed', 'Guilty', 'Disgusted','Embarrassed','Devastated', 'Disappointed', 'Jealous', 'Neutral']
    checks={'q201':'Neutral', 'q202':'Neutral'}  
    visualize=True
    NDE_output, orderedemos=main(ndefile, orderedemos, checks, visualize=visualize)
    with open(os.path.join(rootdir, 'results', 'NDE_output.pkl'), 'wb') as output:
        pickler = pickle.Pickler(output, pickle.HIGHEST_PROTOCOL)
        pickler.dump(NDE_output)
    