"""
Apply all text models (except the MLP model) on tfidf embedding vectors (with folds)

* First define labels in 'model_function' file

"""

import random

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.utils import class_weight

from model_functions import *

random.seed(42)
np.random.seed(42)

# Define models
rf = RandomForestClassifier(random_state=42)
et = ExtraTreesClassifier(random_state=42)
knn = KNeighborsClassifier()
svc = SVC(random_state=42)
rg = LogisticRegression(solver = 'lbfgs', random_state=42)
bayes = MultinomialNB()

vote = VotingClassifier(estimators=[('Random Forests', rf), ('Extra Trees', et), ('KNeighbors', knn), ('SVC', svc),
                                    ('Ridge Classifier', rg), ('bayes', bayes)], voting='hard')

multilayer = MLPClassifier(hidden_layer_sizes=(2140,100, 50, 2), activation="tanh", max_iter=100, alpha=0.0001, verbose=True)

clf_array = [rf, et, knn, svc, rg]

#
rf_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', rf)])
extrt_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', et)])
knn_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', knn)])
svc_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', svc)])
rg_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', rg)])
bs_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', bayes)])
mlp = Pipeline([('tfidf', TfidfVectorizer()), ('clf', multilayer)])
eclf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', vote)])

col = ['path', 'label', 'Type and content']
if second:
    file = "merav/merav2-names.csv"
elif three:
    file = "merav/merav3-names.csv"
else:
    file = "merav/merav_names_final.csv"
if three or second:
    df = pd.read_csv(f'../data/{file}', usecols=col, encoding="ISO-8859-8")
else:
    df = pd.read_csv(f'../data/{file}', usecols=col)

if three:
    df['label_id'] = df['label'].replace({'zero': 0, 'two': 1, 'four': 2})
else:
    df['label_id'] = 1 - df['label'].factorize()[0]
print(df.head())

text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('voting', eclf)])

arr = [rf_clf, extrt_clf, knn_clf, svc_clf, rg_clf, bs_clf, eclf]


data_len = df.shape[0]
no_of_k = 5

kf = KFold(n_splits=no_of_k, shuffle=True, random_state=42)
X = df["Type and content"]
Y = df["label_id"]

for text_clf in arr:
    acc_score_f = 0

    j = 0
    print("clf_is", text_clf['clf'])

    if three:
        models = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        all_l = []
        all_pred = []
    else:
        models = [[0, 0], [0, 0]]
    for train, test in kf.split(X, Y):
        acc_score = []
        weights = class_weight.compute_sample_weight(class_weight='balanced',
                                                     y=Y[train]
                                                     )
        trainData = X[train.astype(int)]
        testData = X[test.astype(int)]
        trainLabels = Y[train.astype(int)]
        testLabels = Y[test.astype(int)]
        print(trainData.shape)
        try:
            text_clf.fit(trainData, trainLabels, clf__sample_weight=weights)
        except:
            text_clf.fit(trainData, trainLabels)
        y_pred = text_clf.predict(testData)
        if three:
            labels = [0, 1, 2]
        else:
            labels = [0, 1]
        cm = confusion_matrix(Y[test], y_pred, labels=labels)
        models += cm
        print(pd.DataFrame(cm, index=labels, columns=labels))
        acc = accuracy_score(Y[test], y_pred)

        if j == 0:
            all_l = Y[test]
            all_pred = y_pred
        else:
            all_l = [*all_l, *Y[test]]
            all_pred = [*all_pred, *y_pred]

        print(f"fold {j} = {acc}")

        j += 1

        acc_score_f += acc
    print(models)
    print(accuracy_score(all_l, all_pred))
    print(classification_report(all_l, all_pred))
    avg_acc_score_f = acc_score_f / 5.0
    print('Avg accuracy for module : {}'.format(avg_acc_score_f), text_clf)
print('data_len: {}'.format(data_len))
