"""
Text & Audio Model
Audio from no fine-tuned model.
* First define audio model and labels in 'model_function' file
"""

import random

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.model_selection import KFold

from keras.layers import Dense, Dropout, concatenate
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras import Input
from keras.utils.np_utils import to_categorical

import tensorflow as tf

from model_functions import *

# Setting the seed for python random numbers
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Settings
num_of_steps = 100

# type of text vector
tfidf = False
bag_of_words = False
# type of audio vectors
isRandom = False  # random audio vector
em_vec = False  # vector of length 8 emotions

if tfidf:
    text_len = 55
else:
    text_len = 768

if MODEL is ModelName.EMOTION:
    audio_len = 1024


elif MODEL is ModelName.BASE:
    audio_len = 768
    if second:
        path = "../data/merav_base_without_ft.csv"

    elif three:
        path = "../data/merav3_base_without_ft.csv"

    else:
        path = "../data/merav2_base_without_ft.csv"

elif MODEL == ModelName.FSFM:
    audio_len = 193
    if second:
        path = "../data/merav_with_fsfm.csv"

    elif three:
        path = "../data/merav3_with_fsfm.csv"

    else:
        path = "../data/merav2_with_fsfm.csv"


elif MODEL is ModelName.EIGHT_EMOTIONS:
    audio_len = 8
    em_vec = True
    if second:
        path = "../data/last_8_vec_emotion.csv"

    elif three:
        path = "../data/last3_8_vec_emotion.csv"

    else:
        path = "../data/last2_8_vec_emotion.csv"

berts_to_append = []


def NN_merge_model(len=text_len):
    # first input
    visible = Input(shape=(audio_len,))
    # second input
    visible2 = Input(shape=(len,))

    do0 = Dropout(0.3)(visible)
    extract1 = Dense(100, activation='tanh')(do0)
    do1 = Dropout(0.3)(extract1)
    print(berts_to_append)
    merge = concatenate([do1, visible2])

    extract2 = Dense(text_len + 100, input_shape=(text_len + 100,), activation='tanh', )(merge)
    extract3 = Dense(100, activation='tanh', )(extract2)
    extract4 = Dense(50, activation='tanh', )(extract3)
    if three:
        output = Dense(3, activation='softmax')(extract4)
    else:
        output = Dense(1, activation='sigmoid')(extract4)
    model = Model(inputs=[visible, visible2], outputs=output)

    print(model.summary())
    # # plot graph
    plot_model(model, to_file='../NN_merge_model.png', show_shapes=True, show_layer_names=True)
    if three:
        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      metrics=['accuracy'])
    else:
        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      metrics=['accuracy'])
    return model


def get_results(model, X, Y, berts, test):
    print('Confusion Matrix')
    if isRandom:
        t = X
    else:
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


def train_and_predict(df):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    X = [item for item in df.vec]
    berts = [item for item in df.bert]
    Y = df.label
    lb = LabelEncoder()
    if three:
        Y = to_categorical(lb.fit_transform(Y))

    cvscores = []
    if three:
        models = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        all_l = []
        all_pred = []
    else:
        models = [[0, 0], [0, 0]]
    accs = []
    losses = []
    fig, ax = plt.subplots(5)
    k = 0
    first = True
    for train, test in kfold.split(X, Y):
        trainData = np.asarray(X)[train.astype(int)]
        testData = np.asarray(X)[test.astype(int)]
        trainLabels = np.asarray(Y)[train.astype(int)]
        testLabels = np.asarray(Y)[test.astype(int)]
        if isRandom:
            randomlist_tr = np.random.normal(0, 0.3, [len(trainData), audio_len])
            randomlist_ts = np.random.normal(0, 0.3, [len(testData), audio_len])
        if tfidf:
            tfidfconverter = CountVectorizer()  # BAG OF WORDS
            tfidfconverter.fit(df.ASR[train])
            tf_tr = tfidfconverter.transform(df.ASR[train]).toarray()
            tf_ts = tfidfconverter.transform(df.ASR[test]).toarray()
            model = NN_merge_model(len(tf_tr[0]))  # send the length of tfidf vector
        else:
            model = NN_merge_model()
        model.reset_states()
        # Calculate the weights for each class so that we can balance the data
        weights = class_weight.compute_class_weight(class_weight='balanced',
                                                    classes=np.unique(df.label),
                                                    y=df.label[train])
        weights = {i: weights[i] for i in range(len(np.unique(df.label)))}

        for i in range(num_of_steps):
            berts_to_append = np.asarray(berts)[train.astype(int)]
            if isRandom and not tfidf:
                history = model.fit([randomlist_tr, berts_to_append], trainLabels, verbose=0, class_weight=weights)
            elif isRandom:
                history = model.fit([randomlist_tr, tf_tr], trainLabels, verbose=0, class_weight=weights)
            elif tfidf:
                history = model.fit([trainData, tf_tr], trainLabels, verbose=0, class_weight=weights)
            else:
                history = model.fit([trainData, berts_to_append], trainLabels, verbose=0, class_weight=weights)

            print(i, "acc: ", history.history['accuracy'], "loss: ", history.history['loss'])
            if float(history.history['accuracy'][0]) > 0.999:
                break
            if i % 10 == 0:
                berts_to_append = np.asarray(berts)[test.astype(int)]
                if isRandom and not tfidf:
                    loss, acc = model.evaluate([randomlist_ts, berts_to_append], testLabels, verbose=0)
                elif isRandom:
                    loss, acc = model.evaluate([randomlist_ts, tf_ts], testLabels, verbose=0)
                elif tfidf:
                    loss, acc = model.evaluate([testData, tf_ts], testLabels, verbose=0)
                else:
                    loss, acc = model.evaluate([testData, berts_to_append], testLabels, verbose=0)
                print(str(i), ": loss = ", loss, " accuracy=", acc)
                losses.append(loss)
                accs.append(acc)
        print("i=", k)
        ax[k].plot(losses, label='loss')
        ax[k].plot(accs, label='acc')
        k = k + 1
        berts_to_append = np.asarray(berts)[test.astype(int)]
        if isRandom and not tfidf:
            scores = model.evaluate([randomlist_ts, berts_to_append], testLabels, verbose=1)
        elif isRandom:
            scores = model.evaluate([randomlist_ts, tf_ts], testLabels, verbose=1)
        elif tfidf:
            scores = model.evaluate([testData, tf_ts], testLabels, verbose=1)
        else:
            scores = model.evaluate([testData, berts_to_append], testLabels, verbose=1)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)
        if isRandom and not tfidf:
            models += get_results(model, randomlist_ts, Y, berts, test)
        elif isRandom:
            models += get_results(model, randomlist_ts, Y, tf_ts, test)
        elif tfidf:
            models += get_results(model, X, Y, tf_ts, test)
        else:
            models += get_results(model, X, Y, berts, test)
        if isRandom and not tfidf:
            y_pred = model.predict([randomlist_ts, berts_to_append])
        elif isRandom:
            y_pred = model.predict([randomlist_ts, tf_ts])
        elif tfidf:
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
    plt.legend()
    plt.show()


def main():
    print(path)
    if three:
        df = pd.read_csv(path, delimiter=',', encoding="ISO-8859-8")
    else:
        df = pd.read_csv(path, delimiter=',')
        df.rename(columns={'asr': 'bert'}, inplace=True)

    if three:
        df['label'].replace({'zero': 0, 'two': 1, 'four': 2}, inplace=True)
    elif second:
        df['label'].replace({'zero': 0, 'four': 1}, inplace=True)
    else:
        df['label'].replace({'zero': 0, 'two': 1}, inplace=True)

    vecs = []
    berts = []
    df_bert = pd.read_csv(path, delimiter=',', usecols=['bert'])
    for row, b in zip(df.iterrows(), df_bert.iterrows()):
        # Convert the vector to list (from string representation)
        row[1]['vec'] = row[1]['vec'].replace('array(', '')
        row[1]['vec'] = row[1]['vec'].replace('\n', '')
        row[1]['vec'] = row[1]['vec'].replace('\r', '')
        row[1]['vec'] = row[1]['vec'].replace(')', '')
        row[1]['vec'] = row[1]['vec'].replace('[', '')
        row[1]['vec'] = row[1]['vec'].replace(']', '')

        temp = np.array(row[1]['vec'].split(', '), dtype=np.float)
        # Append the bag of words vector:
        b[1]['bert'] = b[1]['bert'].replace('[', '')
        b[1]['bert'] = b[1]['bert'].replace(']', '')
        temp_bert = np.array(b[1]['bert'].split(), dtype=np.float)

        vecs.append(temp)
        berts.append(temp_bert)

    df['vec'] = vecs
    df['bert'] = berts

    print(df.head())
    train_and_predict(df)


if __name__ == "__main__":
    main()
