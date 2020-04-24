
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def countdf(df):
    '''
    Feature engineering using the Count CountVectorizer method
    Input: training set
    Output: sparse matrix
    '''
    # create an instance of the CountVectorizer
    countvec = CountVectorizer(ngram_range=(1,2), stop_words='english', binary=True)

    # Create a new variable that is the text column of the df
    text_data = pd.DataFrame(df['review'])

    # Fit and transform the data
    text_cv = countvec.fit(text_data.review)

    # Transformation will be done outside the function because the test data will use the same vectorizer

    return countvec, text_cv

return countdf