import load
import w2v
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import average_precision_score
import pickle



def svm(train_X, val_X, train_y, val_y, max_iter=700, C=1):

    train_y = np.array(train_y).reshape((-1,))
    val_y = np.array(val_y).reshape((-1,))

    model = SVC(C=C, max_iter=max_iter, gamma='auto', probability=True).fit(train_X, train_y)

    print('Train acc = ', model.score(train_X, train_y))
    print('Val acc = ', model.score(val_X, val_y))


    # train_preds = model.predict_proba(train_X)[:,1]
    # print(train_preds[:10])
    # val_preds = model.predict_proba(val_X)[:,1]

    # print('Train avg precision = ', average_precision_score(train_y, train_preds))
    # print('Val avg precision = ', average_precision_score(val_y, val_preds))

    return model

# train_X, val_X, train_y, val_y = load.read_data()

# train_X = w2v.transform(train_X)
# pickle.dump(train_X, open('train_transformed.p', 'wb'))
# val_X = w2v.transform(val_X)
# # pickle.dump(val_X, open('val_transformed.p', 'wb'))

# train_X = pickle.load(open('train_transformed.p', 'rb'))
# val_X = pickle.load(open('val_transformed.p', 'rb'))