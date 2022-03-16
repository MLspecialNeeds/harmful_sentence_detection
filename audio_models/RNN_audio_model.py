"""
RNN model with mfcc as the input.

* First define labels in 'model_function' file
"""

import os
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.applications.densenet import layers
from tensorflow.python.keras.layers import Dropout

from keras.utils.np_utils import to_categorical

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from model_functions import *

SEED = 42
create_mfcc = False  # on the first running the model on the data should be 'True'
frames = 15
first_frame = 30
lstm_size = 15
mfccs = 20  # upto 26!
num_of_steps = 100

tf.random.set_seed(SEED)
np.random.seed(SEED)

if second:
    path_df = "../data/merav_all.csv"
    mfcc_path = "../data/all.txt"
    langs = ['0', '4']
elif three:
    path_df = "../data/merav3-all-mfcc.csv"
    mfcc_path = "../data/all3.txt"
    langs = ['0', '2', '4']
else:
    path_df = "../data/merav2-all.csv"
    mfcc_path = "../data/all.txt"
    langs = ['0', '2']


def get_results(model, X, Y, test):
    print('Confusion Matrix')
    t = np.asarray(X)[test]
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


def build_model():
    model = keras.Sequential()
    model.add(layers.LSTM(lstm_size, activation='relu'))
    model.add(Dropout(0.3))
    if three:
        model.add(layers.Dense(3, activation='softmax'))
    else:
        model.add(layers.Dense(1, activation='sigmoid'))
    if three:
        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      metrics=['accuracy'])
    else:
        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      metrics=['accuracy'])

    return model


def train_and_predict(df, all):
    kfold = KFold(n_splits=5, random_state=SEED, shuffle=True)
    cvscores = []
    if three:
        models = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        all_l = []
        all_pred = []

    else:
        models = [[0, 0], [0, 0]]
    X = [item[0] for item in all]
    Y = [[item[1]] for item in all]
    # Hot encoding y
    lb = LabelEncoder()
    if three:
        Y = to_categorical(lb.fit_transform(Y))
    first = True
    print(Y)
    for train, test in kfold.split(X, Y):
        trainData = np.asarray(X)[train.astype(int)]
        testData = np.asarray(X)[test.astype(int)]
        trainLabels = np.asarray(Y)[train.astype(int)]
        testLabels = np.asarray(Y)[test.astype(int)]
        model = build_model()

        model.reset_states()
        # model.summary()
        weights = class_weight.compute_class_weight(class_weight='balanced',
                                                    classes=np.unique(df.label_num),
                                                    y=df.label_num[train])
        weights = {i: weights[i] for i in range(len(langs))}
        print(weights)
        for i in range(num_of_steps):
            history = model.fit(trainData, trainLabels, verbose=0, class_weight=weights, )
            print(i, "acc: ", history.history['accuracy'], "loss: ", history.history['loss'])
            if float(history.history['accuracy'][0]) > 0.9999:
                break
            if i % 10 == 0:
                loss, acc = model.evaluate(testData, testLabels, verbose=0)
                print(str(i), ": loss = ", loss, " accuracy=", acc)

        scores = model.evaluate(testData, testLabels, verbose=1)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)
        models += get_results(model, X, Y, test)
        y_pred = model.predict(testData)

        if three:
            if first:
                all_l = np.argmax(testLabels, axis=1)
                all_pred = np.argmax(y_pred, axis=1)
            else:
                all_l = [*all_l, *np.argmax(testLabels, axis=1)]
                all_pred = [*all_pred, *np.argmax(y_pred, axis=1)]

            first = False

        print(models)
        if three:
            print(classification_report(all_l, all_pred))
        else:
            TP = models[1][1]
            FP = models[0][1]
            FN = models[1][0]
            TN = models[0][0]
            print(f"Accuracy: {(TP + TN) / (TP + TN + FP + FN)}")
            print(f"Precision: {(TP) / (TP + FP)}")
            print(f"Recall: {TP / (TP + FN)}")
            print(f"F1_score: {(2 * TP / (TP + FN) * (TP) / (TP + FP)) / (TP / (TP + FN) + (TP) / (TP + FP))}\n\n")
            print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


def main():
    if create_mfcc:
        df = pd.read_csv(path_df, sep=',')
        labels = []
        all = []
        for row in df.iterrows():
            (rate, sig) = wav.read(row[1]['path'])
            mfcc_feat = mfcc(sig, rate)
            d_mfcc_feat = delta(mfcc_feat, 2)
            curr = logfbank(sig, rate)
            frame = curr[first_frame:(first_frame + frames), 0:mfccs] / 20

            if len(frame) >= frames:
                if three:
                    if row[1]['path'][0] == '0':
                        label_num = 0
                    elif row[1]['path'][0] == '2':
                        label_num = 1
                    elif row[1]['path'][0] == '4':
                        label_num = 2
                else:
                    if row[1]['path'][0] == '0':
                        label_num = 0
                    else:
                        label_num = 1
                all.append((frame, 1.0 * label_num))
                labels.append(label_num)
            with open(mfcc_path, "wb") as fp:  # Pickling
                pickle.dump(all, fp)
        df["mfcc"] = all
        df["label_num"] = labels
        df.to_csv(path_df, index=False)
    else:
        print(path_df)
        df = pd.read_csv(path_df, delimiter=',')
        with open(mfcc_path, "rb") as fp:  # Unpickling
            all = pickle.load(fp)
        berts = []
        for row in df.iterrows():
            row[1]['bert'] = row[1]['bert'].replace('[', '')
            row[1]['bert'] = row[1]['bert'].replace(']', '')
            row[1]['bert'] = row[1]['bert'].replace('\n', '')
            berts.append(np.array(row[1]['bert'].split(), dtype=np.float))
        df["bert"] = berts
    print(df.mfcc)
    train_and_predict(df, all)


if __name__ == "__main__":
    main()
