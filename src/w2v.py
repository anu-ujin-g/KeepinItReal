import gensim.downloader
import pandas as pd
import numpy as np


def w2v(df):
    w2v = gensim.downloader.load('word2vec-google-news-300')

    # Remove punctuation
    df['review'] = df['review'].apply(lambda x: ''.join([c for c in x if c.isalpha() or c==' ']))

    # Remove words that have no vector representations
    df['review'] = df['review'].apply(lambda x: ' '.join(word for word in x.split() if word in model.vocab))


    w2v = []
    for idx, row in df.iterrows():
        if row.review == '':
            w2v.append(np.zeros(300))
        else:
            w2v.append(np.mean([w2v.word_vec(word) for word in row.review.split()]))
    w2v = np.array(w2v)



    return w2v