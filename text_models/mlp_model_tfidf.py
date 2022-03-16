"""
MLP model on TFIDF text embedding.

* First define labels in 'model_function' file
"""

import random


from keras.utils.np_utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder

from keras.layers import Dense, Dropout
from keras.models import Sequential
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from keras.utils.vis_utils import plot_model
from sklearn.utils import class_weight

from model_functions import *

# Setting the seed for python random numbers
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

bag_of_words = False
num_of_steps = 100

col = ['path', 'label', 'Type and content']
if second:
    file = "merav/merav2-names.csv"
elif three:
    file = "merav/merav3-names.csv"
else:
    file = "merav/merav_names_final.csv"



def NN_model(len):
    # Building our model
    model = Sequential()
    model.add(Dense(len, activation='tanh'))
    model.add(Dense(100, activation='tanh'))
    model.add(Dense(50, activation='tanh'))
    if three:
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      metrics=['accuracy'])
    else:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      metrics=['accuracy'])
    plot_model(model, to_file='../first_model.png', show_shapes=True, show_layer_names=True)
    return model


def train_and_predict(df):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    j = 0
    acc_score_f = 0
    if three:
        models = [[0,0,0],[0,0,0],[0,0,0]]
    else:
        models = [[0,0],[0,0]]

    for train, test in kfold.split(df.ASR, df.label_id):
        if bag_of_words:
            tfidfconverter = CountVectorizer()
        else:
            tfidfconverter = TfidfVectorizer()
        tfidfconverter.fit(df.ASR[train])
        X_tr = tfidfconverter.transform(df.ASR[train]).toarray()
        X_ts = tfidfconverter.transform(df.ASR[test]).toarray()
        Y = df.label_id
        lb = LabelEncoder()
        if three:
            Y = to_categorical(lb.fit_transform(Y))
        y_tr = Y[train]
        y_ts = Y[test]
        # Calculate the weights for each class so that we can balance the data
        weights = class_weight.compute_sample_weight(class_weight='balanced',
                                                     y=df.label_id[train]
                                                     )
        weights = {i: weights[i] for i in range(len(np.unique(df.label)))}

        model = NN_model(X_tr.shape[1])
        model.reset_states()
        for i in range(num_of_steps):
            history = model.fit(X_tr, y_tr, verbose=0,class_weight=weights)
            if i == 0:
                model.summary()
            print(i, "acc: ", history.history['accuracy'], "loss: ", history.history['loss'])
            if float(history.history['accuracy'][0]) > 0.9999:
                break
            if i % 10 == 0:
                loss, acc = model.evaluate(X_ts, y_ts, verbose=0)
                print(str(i), ": loss = ", loss, " accuracy=", acc)

        y_pred = model.predict(X_ts)

        if three:
            labels = [0,1,2]
            acc = accuracy_score(np.argmax(y_ts, axis=1), np.argmax(y_pred, axis=1))
            cm = confusion_matrix(np.argmax(y_ts, axis=1), np.argmax(y_pred, axis=1), labels=labels)
        else:
            labels = [0,1]
            acc = accuracy_score(y_ts, np.rint(y_pred))
            cm = confusion_matrix(y_ts, np.rint(y_pred), labels=labels)

        models += cm
        print(pd.DataFrame(cm, index=labels, columns=labels))
        print(f"fold {j} = {acc}")

        j += 1
        acc_score_f += acc

    print(models)
    if not three:
        TP = models[1][1]
        FP = models[0][1]
        FN = models[1][0]
        TN = models[0][0]
        print(f"Accuracy: {(TP + TN) / (TP + TN + FP + FN)}")
        print(f"Precision: {(TP) / (TP + FP)}")
        print(f"Recall: {TP / (TP + FN)}")
        print(f"F1_score: {(2 * TP / (TP + FN) * (TP) / (TP + FP)) / (TP / (TP + FN) + (TP) / (TP + FP))}\n\n")
        print("%.2f%% (+/- %.2f%%)" % (np.mean(acc_score_f), np.std(acc_score_f)))
    avg_acc_score_f = acc_score_f / 5

    print('Avg accuracy for module : {}'.format(avg_acc_score_f))


def main():
    print(file)
    if three or second:
        df = pd.read_csv(f'data/{file}', delimiter=',', usecols=col, encoding="ISO-8859-8")
        df = df.rename(columns={'Type and content': 'ASR'})
    else:
        df = pd.read_csv(f'data/{file}', delimiter=',', usecols=col)
        df = df.rename(columns={'Type and content': 'ASR'})
    if three:
        df['label_id'] = df['label'].replace({'zero': 0, 'two': 1, 'four': 2})
    else:
        df['label_id'] = 1 - df['label'].factorize()[0]

    print(df.head())
    train_and_predict(df)

if __name__ == "__main__":
    main()
