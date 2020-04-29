import gensim.downloader
import pandas as pd
import numpy as np


def transform(df):
    w2v = gensim.downloader.load('word2vec-google-news-300')

    df = df.dropna()
    # Remove punctuation
    df['review'] = df['review'].apply(lambda x: ''.join([c for c in x if c.isalpha() or c==' ']))

    # Remove words that have no vector representations
    df['review'] = df['review'].apply(lambda x: ' '.join(word for word in x.split() if word in w2v.vocab))


    res = []
    for idx, row in df.iterrows():
        if row.review == '':
            res.append(np.zeros(300))
        else:
            res.append(np.mean([w2v.word_vec(word) for word in row.review.split()], axis=0))
    res = np.array(res)



    return w2v, res