import os
import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


def clean_data(df):
    '''
    Cleaning the text of the data and adding a new column for length of review
    '''

    # remove stop words
    stop = stopwords.words('english')
    df['review'] = df['review'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop))

    # remove all punctuations
    tokenizer = RegexpTokenizer(r'\w+')
    df['review'] = df['review'].apply(lambda x: ' '.join(word for word in tokenizer.tokenize(x)))

    # make the words lowercase
    df['review'] = df['review'].apply(lambda x: x.lower())
    

    print("Data Cleaning Complete")
    return df


