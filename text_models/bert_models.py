"""
Apply all text models (except the MLP model) on BERT embedding vectors with folds

* First define labels in 'model_function' file

"""
import random

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
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
vote = VotingClassifier(estimators=[('Random Forest', rf), ('Extra Trees', et), ('KNeighbors', knn), ('SVC', svc),
                                    ('Ridge Classifier', rg), ])
classifiers = [rf, et, knn, svc, rg, vote]


def get_results(model, t, Y, test):
    print('Confusion Matrix')
    l = np.asarray(Y)[test]
    y_pred = model.predict(t)
    if three:
        labels = [0, 1, 2]
        y_pred = np.argmax(y_pred, axis=1)
        l = np.argmax(l, axis=1)
    else:
        labels = [0, 1]
        y_pred = np.rint(y_pred)

    cm = confusion_matrix(l, y_pred, labels=labels)
    print(pd.DataFrame(cm, index=labels, columns=labels))
    if three:
        print(classification_report(l, y_pred))
    else:
        Y_test = np.asarray(Y)[test]
        from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
        print(f"Accuracy: {round(accuracy_score(Y_test, y_pred), 2)}")
        print(f"Precision: {round(precision_score(Y_test, y_pred), 2)}")
        print(f"Recall: {round(recall_score(Y_test, y_pred), 2)}")
        print(f"F1_score: {round(f1_score(Y_test, y_pred), 2)}\n\n")
    return cm


# Preprocess data
if second:
    file = "merav_base_without_ft.csv"
elif three:
    file = "merav3_base_without_ft.csv"
else:
    file = "merav2_base_without_ft.csv"

col = ['path', 'label', 'bert']

if three:
    df = pd.read_csv(f'../data/{file}', usecols=col, encoding="ISO-8859-8")
    df['label_id'] = df['label'].replace({'zero': 0, 'two': 1, 'four': 2})
else:
    df = pd.read_csv(f'../data/{file}', usecols=col)
    df['label_id'] = 1 - df['label'].factorize()[0]
print(df.head())

vecs = []
for row in df.iterrows():
    row[1]['bert'] = row[1]['bert'].replace('[', '')
    row[1]['bert'] = row[1]['bert'].replace(']', '')
    temp = np.array(row[1]['bert'].split(), dtype=np.float)

    vecs.append(temp)
df['bert'] = vecs
print(df.head())

# Apply models with 5 fold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for classifier in classifiers:
    first = True

    if three:
        models = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        all_l = []
        all_pred = []
    else:
        models = [[0, 0], [0, 0]]

    print(
        f'-------------------------------------------------------{classifier}-------------------------------------------------------')
    j = 0
    acc_score_f = 0
    for train, test in kfold.split(df.bert, df.label_id):

        # Calculate the weights for each class so that we can balance the data
        weights = class_weight.compute_sample_weight(class_weight='balanced',
                                                     y=df.label_id[train]
                                                     )
        X = [item for item in df.bert]
        Y = df.label_id
        X_tr = np.asarray(X)[train]
        X_ts = np.asarray(X)[test]
        try:
            classifier.fit(X_tr, Y[train], sample_weight=weights)
        except:
            classifier.fit(X_tr, Y[train])

        y_pred = classifier.predict(X_ts)
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

    print('Avg accuracy for module : {}'.format(avg_acc_score_f), classifier)
