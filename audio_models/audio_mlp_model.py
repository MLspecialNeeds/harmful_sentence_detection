"""
Audio / Audio with Text Model with no reducing of the audio vector.
Audio from no fine-tuned model.
Text - Bert
Also can be used to train text only by define - flag_vec = False

* First define audio model and labels in 'model_function' file
"""



import random

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.model_selection import KFold

from keras.layers import Dense
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import plot_model


import tensorflow as tf
from model_functions import *

# Setting the seed for python random numbers
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

num_of_steps = 100
isRandom = False  # Random vector
# text or audio
flag_bag = True
flag_vec = False


if MODEL is ModelName.EIGHT_EMOTIONS:
    em_vec = True
else:
    em_vec = False

if MODEL is ModelName.EMOTION:
    if second:
        file = "merav2_emotion_without_ft.csv"
    elif three:
        file = "merav3_emotion_without_ft.csv"
    else:
        file = "merav_em_without_ft.csv"

elif MODEL is ModelName.BASE:
    file = "merav/merav_base_without_ft.csv"
    if second:
        file = "merav/merav2_base_without_ft.csv"
    elif three:
        file = "merav/merav3_base_without_ft.csv"

elif MODEL is ModelName.EIGHT_EMOTIONS:
        if second:
            file = "last2_8_vec_emotion.csv"
        elif three:
            file = "last3_8_vec_emotion.csv"
        else:
            file = "last_8_vec_emotion.csv"

# Audio and text concentrate
if flag_bag and flag_vec:
    if MODEL is ModelName.EIGHT_EMOTIONS:
        vec_len = 776
    elif MODEL is ModelName.EMOTION:
            vec_len = 1792
    else: #BASE
            vec_len = 823
# Audio only
elif flag_vec and not flag_bag:
    if MODEL is ModelName.EIGHT_EMOTIONS:
        vec_len = 8
    elif MODEL is ModelName.EMOTION:
        vec_len = 1024
    elif MODEL is ModelName.BASE:
        vec_len = 768

if isRandom:
    vec_len = 768


def NN_model():
    # Building our model
    if em_vec: # Model for vector length 8
        model = Sequential()
        model.add(Dense(vec_len, input_shape=(vec_len,), activation='tanh'))
        model.add(Dense(4, input_shape=(vec_len,), activation='tanh'))
        model.add(Dense(4, activation='tanh'))

    else:
        model = Sequential()
        model.add(Dense(vec_len, input_shape=(vec_len,), activation='tanh'))
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


def get_results(model, X, Y, test):
    print('Confusion Matrix')
    if isRandom:
        t = X
    else:
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
    print(classification_report(l, y_pred))
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
            randomlist_tr = np.random.normal(0, 0.3, [len(trainData), vec_len])
            randomlist_ts = np.random.normal(0, 0.3, [len(testData), vec_len])

        model = NN_model()
        model.reset_states()
        model.summary()
        # Calculate the weights for each class so that we can balance the data
        weights = class_weight.compute_class_weight(class_weight='balanced',
                                                    classes=np.unique(df.label),
                                                    y=df.label[train])
        weights = {i: weights[i] for i in range(len(np.unique(df.label)))}
        for i in range(num_of_steps):
            if isRandom:
                history = model.fit(randomlist_tr, trainLabels, verbose=0, class_weight=weights)
            else:
                history = model.fit(trainData, trainLabels, verbose=0, class_weight=weights)
            print(i, "acc: ", history.history['accuracy'], "loss: ", history.history['loss'])
            if float(history.history['accuracy'][0]) > 0.9999:
                break
            if i % 10 == 0:
                if isRandom:
                    loss, acc = model.evaluate(randomlist_ts, testLabels, verbose=0)
                else:
                    loss, acc = model.evaluate(testData, testLabels, verbose=0)
                print(str(i), ": loss = ", loss, " accuracy=", acc)
                losses.append(loss)
                accs.append(acc)
        print("i=", k)
        ax[k].plot(losses, label='loss')
        ax[k].plot(accs, label='acc')
        k = k + 1
        if isRandom:
            scores = model.evaluate(randomlist_ts, testLabels)
        else:
            scores = model.evaluate(testData, testLabels, verbose=1)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)
        if isRandom:
            models += get_results(model, randomlist_ts, Y, test)
        else:
            models += get_results(model, X, Y, test)
        if isRandom:
            y_pred = model.predict(randomlist_ts)
        else:
            y_pred = model.predict(testData)
        if three:
            if first:
                all_l = np.argmax(testLabels, axis=1)
                all_pred = np.argmax(y_pred, axis=1)
            else:
                all_l = [*all_l, *np.argmax(testLabels, axis=1)]
                all_pred = [*all_pred, *np.argmax(y_pred, axis=1)]
        else:
            if first:
                all_l = testLabels
                all_pred = np.argmax(y_pred, axis=1)
            else:
                all_l = [*all_l, *testLabels]
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
    print(file)
    if three:
        df = pd.read_csv(f'data/{file}', delimiter=',', encoding="ISO-8859-8")
    else:
        df = pd.read_csv(f'data/{file}', delimiter=',')

    print(df)
    if three:
        df['label'].replace({'zero': 0, 'two': 1, 'four': 2}, inplace=True)
    elif second:
        df['label'].replace({'zero': 0, 'four': 1}, inplace=True)
    else:
        df['label'].replace({'zero': 0, 'two': 1}, inplace=True)
        df = df.rename(columns={'asr': 'bert'})

    vecs = []
    temp = []
    for row in df.iterrows():
        if flag_vec and not isRandom:
            # Convert the vector to list (from string representation)
            row[1]['vec'] = row[1]['vec'].replace('array(', '')
            row[1]['vec'] = row[1]['vec'].replace('\n', '')
            row[1]['vec'] = row[1]['vec'].replace('\r', '')
            row[1]['vec'] = row[1]['vec'].replace(')', '')
            row[1]['vec'] = row[1]['vec'].replace('[', '')
            row[1]['vec'] = row[1]['vec'].replace(']', '')
            if not flag_bag:
                temp = np.array(row[1]['vec'].split(', '), dtype=np.float)
            else:
                a = np.array(row[1]['vec'].split(', '), dtype=np.float)
        if flag_bag:
            # Append the bag of words vector
            row[1]['bert'] = row[1]['bert'].replace('[', '')
            row[1]['bert'] = row[1]['bert'].replace(']', '')
            if not flag_vec:
                temp = np.array(row[1]['bert'].split(', '), dtype=np.float)
            else:
                b = np.array(row[1]['bert'].split(), dtype=np.float)
                temp = np.concatenate((a, b))

        vecs.append(temp)
    df['vec'] = vecs
    print(df.head())
    train_and_predict(df)


if __name__ == "__main__":
    main()
