#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import load
import spacy
import numpy as np
import gensim.downloader
from subprocess import run
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

## progress bar for bert so I don't go crazy...
from tqdm import tqdm

def cv(df):
    '''Returns fitted `cv.fit(df.review)` with `.transform(df.review)` function
    '''
    countvec = CountVectorizer(ngram_range=(1,2), stop_words='english', binary=True)
    
    return countvec.fit(df)


def tfidf(df):
    '''Returns fitted `tfidf.fit(df.review)` with `.transform(df.review)` function
    '''
    tfidfvec = TfidfVectorizer(stop_words='english', binary=True, use_idf=True)

    return tfidfvec.fit(df)


def w2v(df):
    '''Returns w2v transformer with `.transform(df.review)` function
    '''
    # prompts download of 1.6GB model if not already installed
    print("please be patient while w2v is loaded...")
    w2v = gensim.downloader.load('word2vec-google-news-300')

    def transform(df):
        # Remove punctuation
        df = df.apply(lambda x: ''.join([c for c in x if c.isalpha() or c==' ']))
        # Remove words that have no vector representations
        df = df.apply(lambda x: ' '.join(word for word in x.split() if word in w2v.vocab))

        res = []
        for idx, item in df.iteritems():
            if item == '':
                res.append(np.random.randn(300))
            else:
                res.append(np.mean([w2v.word_vec(word) for word in item.split()], axis=0))
        return np.array(res)
    ## .transform(df) casting
    w2v.transform = transform

    return w2v


def bert(df, save_name=False, load_name=False):
    '''Returns bert transformer with `.transform(df.review)` function
    '''
    try:
        # We will use base bert; bert is not retrained on df
        _bert = 'en_trf_bertbaseuncased_lg'
        nlp = spacy.load(_bert)
    except:
        # prompts download of ??GB model if not already installed
        if not run(f'python -m spacy {_bert}').status:
            nlp = spacy.load(_bert)
        else:
            raise Exception(f"can't load: {_bert}")
            
    def transform(df):
        if load_name:
            return load.one_file(load_name)
        
        # Initiate progress bar
        tqdm.pandas()
        df = df.progress_apply(lambda x: nlp(x).vector).to_list()
        
        
        # Save so you don't need to re-transform the dataset every time
        if save_name:
            df.to_pickle(f'{load.get_data_path()}{save_name}_bert.pickle')
        
        return df
    ## .transform(df) casting
    bert.transform = transform
        
    return bert