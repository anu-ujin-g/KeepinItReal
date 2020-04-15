import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf(train, val):
    '''
    Feature engineering using the TF-IDF method
    Input: training and validation set
    Output: sparse matrix
    '''
    tfidfvec = TfidfVectorizer(stop_words='english', binary=True)

    # Create a new variable that is the text column of the df
    text_data = pd.DataFrame(train['review'])
    text_val = pd.DataFrame(val['review'])
    # # Add a new column for review length
    # train['len(review)'] = train['review'].apply(lambda x: len(x.split()))

    # Fit the text data and build a vocabulary
    train_tfidf = tfidfvec.fit(text_data.review)
    train_tfidf = tfidfvec.transform(text_data.review)
    val_tfidf = tfidfvec.transform(text_val.review)

    print("Created a sparse matrix of size:",train_tfidf.shape)
    return train_tfidf, val_tfidf




    # # summarize encoded vector
    # print(tfidf.vocabulary_)
    # print(vectorizer.idf_)
    # print(vector.shape)
    # print(vector.toarray())

