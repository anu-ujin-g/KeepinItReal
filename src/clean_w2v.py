import gensim.downloader
import pandas as pd
import numpy as np
import load

def clean_w2v(X):
    # Removes punctuation and words that w2v doesn't know.
    # Everything else is preserved (including uppercase letters)

    w2v = gensim.downloader.load('word2vec-google-news-300')
    reviews = X['review']


    i = 0
    new_reviews = []
    removed = {}
    for review in reviews:
        new_review = []
        no_punctuation = ''.join([c if c.isalpha() or c.isdigit() or c==' ' else ' ' for c in review])

        for word in no_punctuation.split():
            if word in w2v.vocab or word.lower() in w2v.vocab or word.upper() in w2v.vocab:
                new_review.append(word)
            else:
                if word in removed:
                    removed[word]+=1
                else:
                    removed[word] = 1
        new_reviews.append(new_review)

    X['review'] = list(map(lambda x: ' '.join(x), new_reviews))
    return X
