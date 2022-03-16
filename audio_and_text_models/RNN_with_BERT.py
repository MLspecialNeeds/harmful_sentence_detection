"""
Running audio & text model of mfcc with RNN as the audio and BERT as the text input.

* First define labels in 'model_function' file
"""

import pickle

from keras.models import Model
from keras.layers import Dense, concatenate, Dropout
from keras.utils.np_utils import to_categorical
from keras import Input, layers
from keras.utils.vis_utils import plot_model

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.model_selection import KFold

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank

import scipy.io.wavfile as wav
import tensorflow as tf

from model_functions import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

SEED = 42
frames = 15
first_frame = 30
lstm_size = 128
mfccs = 20

num_of_steps = 100
create_mfcc = False  # on the first running the model on the data should be 'True'
tfidf = False

bert_len = 768
if second:
    langs = ['0', '4']
    path_df = "merav_all.csv"
    mfcc_path = "../data/all.txt" # need to be created
elif three:
    langs = ['0', '2', '4']
    path_df = "merav3-all-mfcc.csv"
    mfcc_path = "../data/all3.txt"
else:  # 0 & 4 data
    langs = ['0', '2']
    path_df = "merav2-all.csv"
    mfcc_path =  "../data/all2.txt" # need to be created

tf.random.set_seed(SEED)
np.random.seed(SEED)


def get_results(model, X, Y, berts, test):
    print('Confusion Matrix')
    t = np.asarray(X)[test]
    if tfidf:
        b = berts
    else:
        b = np.asarray(berts)[test]
    l = np.asarray(Y)[test]
    y_pred = model.predict([t, b])
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


def build_model(len=bert_len):
    # first input
    visible = Input(shape=(15, 20,))
    # second input
    visible2 = Input(shape=(len,))
    rnn0 = layers.LSTM(lstm_size, activation='relu')(visible)
    rnn1 = Dropout(0.3)(rnn0)
    extract1 = Dense(100, activation='tanh')(rnn1)
    do1 = Dropout(0.3)(extract1)
    merge = concatenate([do1, visible2])

    extract2 = Dense(len + 100, input_shape=(len + 100,), activation='tanh')(merge)
    extract3 = Dense(100, activation='tanh')(extract2)
    extract4 = Dense(50, activation='tanh')(extract3)
    if three:
        output = Dense(3, activation='softmax')(extract4)
    else:
        output = Dense(1, activation='sigmoid')(extract4)
    model = Model(inputs=[visible, visible2], outputs=output)
    print(model.summary())
    # # plot graph
    plot_model(model, to_file='../RNN_merge_model.png', show_shapes=True, show_layer_names=True)

    if three:
        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      metrics=['accuracy'])
    else:
        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
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
    berts = [item for item in df.bert]
    print(len(X))
    print(len(berts))
    Y = [item for item in df.label_num]
    # Hot encoding y
    lb = LabelEncoder()
    if three:
        Y = to_categorical(lb.fit_transform(Y))
    first = True
    for train, test in kfold.split(X, Y):
        trainData = np.asarray(X)[train.astype(int)]
        testData = np.asarray(X)[test.astype(int)]
        trainLabels = np.asarray(Y)[train.astype(int)]
        testLabels = np.asarray(Y)[test.astype(int)]
        if tfidf:
            tfidfconverter = TfidfVectorizer()
            tfidfconverter.fit(df.ASR[train])
            tf_tr = tfidfconverter.transform(df.ASR[train]).toarray()
            tf_ts = tfidfconverter.transform(df.ASR[test]).toarray()
            print(tf_tr[0])
            model = build_model(len(tf_tr[0]))  # send the length of tfidf vector
        else:
            model = build_model()
        # model.summary()
        classes = df.label_num
        weights = class_weight.compute_class_weight(class_weight='balanced',
                                                    classes=np.unique(classes),
                                                    y=classes[train])
        weights = {i: weights[i] for i in range(len(langs))}
        print(weights)
        for i in range(num_of_steps):
            berts_to_append = np.asarray(berts)[train.astype(int)]
            if tfidf:
                history = model.fit([trainData, tf_tr], trainLabels, verbose=0, class_weight=weights)
            else:
                history = model.fit([trainData, berts_to_append], trainLabels, verbose=0, class_weight=weights)
            print(i, "acc: ", history.history['accuracy'], "loss: ", history.history['loss'])
            if float(history.history['accuracy'][0]) > 0.9999:
                break
            if i % 10 == 0:
                berts_to_append = np.asarray(berts)[test.astype(int)]
                if tfidf:
                    loss, acc = model.evaluate([testData, tf_ts], testLabels, verbose=0)
                else:
                    loss, acc = model.evaluate([testData, berts_to_append], testLabels, verbose=0)
                print(str(i), ": loss = ", loss, " accuracy=", acc)

        berts_to_append = np.asarray(berts)[test.astype(int)]
        if tfidf:
            scores = model.evaluate([testData, tf_ts], testLabels, verbose=1)
        else:
            scores = model.evaluate([testData, berts_to_append], testLabels, verbose=1)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)
        if tfidf:
            models += get_results(model, X, Y, tf_ts, test)
        else:
            models += get_results(model, X, Y, berts, test)
        if tfidf:
            y_pred = model.predict([testData, tf_ts])
        else:
            y_pred = model.predict([testData, berts_to_append])

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


if __name__ == "__main__":
    if create_mfcc:
        df = pd.read_csv('../data/' + path_df, sep=',')
        labels = []
        all = []
        mfccc = []
        berts = []
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
        df.to_csv('../data/' + path_df, index=False)
    else:
        print(path_df)
        df = pd.read_csv('../data/'+path_df, delimiter=',')
        with open(mfcc_path, "rb") as fp:  # Unpickling
            all = pickle.load(fp)
        berts = []
        mfccs = []
        for row in df.iterrows():
            row[1]['bert'] = row[1]['bert'].replace('[', '')
            row[1]['bert'] = row[1]['bert'].replace(']', '')
            row[1]['bert'] = row[1]['bert'].replace('\n', '')
            berts.append(np.array(row[1]['bert'].split(), dtype=np.float))

        df["bert"] = berts
    print(df.mfcc)
    train_and_predict(df, all)
