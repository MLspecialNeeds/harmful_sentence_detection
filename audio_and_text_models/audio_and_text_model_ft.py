"""
Text & Audio Model
Audio from fine-tuned model (folds) reduced to 100.
Text - Bert
Also can be used to train text only by define - flag_vec = False

* First define audio model and labels in 'model_function' file
"""

from keras import Input, Model
from keras.layers import Dropout, Dense, concatenate
from keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import plot_model

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

import tensorflow as tf

from model_functions import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

tf.random.set_seed(42)
np.random.seed(42)
# Setting the seed for python random numbers
np.random.seed(42)
tf.random.set_seed(42)

# Settings
num_of_steps = 100

# type of text vector
tfidf = False
bag_of_words = False
# type of audio vectors
isRandom = True # random audio vector
em_vec = False  # vector of length 8 emotions

if tfidf :
    text_len = 55
else:
    text_len = 768 # bert

if MODEL is ModelName.EMOTION:
    audio_len = 1024
    if second:
        path = '../models/merav_em_ft_3_2_5_0.4'
    elif three:
        path = '../models/merav3_emotion'
    else:
        path = '../models/merav2_emotion_ft_new_method_best'


elif MODEL is ModelName.BASE:
    audio_len = 768
    if second:
        path = '../models/merav_base_ft_3_2_0.25'
    elif three:
        path = '../models/merav3_base'
    else:
        path = '../models/merav2_base_ft_updated'

def NN_merge_model(len = text_len):
    # first input
    visible = Input(shape=(audio_len,))
    # second input
    visible2 = Input(shape=(len,))

    do0 = Dropout(0.3)(visible)
    extract1 = Dense(100, activation='tanh')(do0)
    do1 = Dropout(0.3)(extract1)
    merge = concatenate([do1, visible2])

    extract2 = Dense(len+100, input_shape=(len+100,), activation='tanh',)(merge)
    extract3 = Dense(100, activation='tanh',)(extract2)
    extract4 = Dense(50, activation='tanh',)(extract3)
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

def get_results(model, X_ts, Y_ts, bert_ts):

    print('Confusion Matrix')
    y_pred = model.predict([X_ts, bert_ts])
    if three:
        labels = [0, 1, 2]
        y_pred = np.argmax(y_pred, axis=1)
    else:
        labels = [0, 1]
        y_pred = np.rint(y_pred)
    cm = confusion_matrix(Y_ts, y_pred, labels=labels)
    print(pd.DataFrame(cm, index=labels, columns=labels))
    if three:
        print(classification_report(Y_ts, y_pred))
    else:
        Y_test = Y_ts
        from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
        print(f"Accuracy: {round(accuracy_score(Y_test, y_pred), 2)}")
        print(f"Precision: {round(precision_score(Y_test, y_pred), 2)}")
        print(f"Recall: {round(recall_score(Y_test, y_pred), 2)}")
        print(f"F1_score: {round(f1_score(Y_test, y_pred), 2)}\n\n")
    return cm


def train_and_predict(train_df, test_df):
    X_tr = np.asarray([item for item in train_df.vec])
    X_ts = np.asarray([item for item in test_df.vec])
    Y_tr = np.asarray([item for item in train_df.label])
    Y_ts = np.asarray([item for item in test_df.label])
    lb = LabelEncoder()
    if three:
        Y_tr = to_categorical(lb.fit_transform(Y_tr))
        Y_ts = to_categorical(lb.fit_transform(Y_ts))
    bert_tr = np.asarray([item for item in train_df.bert])
    bert_ts = np.asarray([item for item in test_df.bert])
    randomlist_ts = []
    if isRandom:
        randomlist_tr = np.random.normal(0, 0.3, [len(X_tr), audio_len])
        randomlist_ts = np.random.normal(0, 0.3, [len(X_ts), audio_len])

    if tfidf:
        tfidfconverter = CountVectorizer()  # BAG OF WORDS
        tfidfconverter.fit(train_df.AS)
        tf_tr = tfidfconverter.transform(train_df.ASR).toarray()
        print(tf_tr[0])
        tf_ts = tfidfconverter.transform(test_df.ASR).toarray()
        model = NN_merge_model(len(tf_tr[0])) # send the length of tfidf vector
    else:
        model = NN_merge_model()

    model.reset_states()
    model.summary()

    # Calculate the weights for each class so that we can balance the data
    weights = class_weight.compute_class_weight(class_weight='balanced',
                                                classes=np.unique(train_df.label),
                                                y=train_df.label)
    weights = {i: weights[i] for i in range(len(np.unique(train_df.label)))}

    for i in range(num_of_steps):
        if isRandom and not tfidf:
            history = model.fit([randomlist_tr, bert_tr], Y_tr, verbose=0, class_weight=weights)
        elif isRandom:
            history = model.fit([randomlist_tr, tf_tr], Y_tr, verbose=0, class_weight=weights)
        elif tfidf:
            history = model.fit([X_tr, tf_tr], Y_tr, verbose=0, class_weight=weights)
        else:
            history = model.fit([X_tr, bert_tr], Y_tr, verbose=0, class_weight=weights, )
        print(i, "acc: ", history.history['accuracy'], "loss: ", history.history['loss'])
        if float(history.history['accuracy'][0]) > 0.9999:
            break
        if i % 10 == 0:
            if isRandom and not tfidf:
                loss, acc = model.evaluate([randomlist_ts, bert_ts], Y_ts, verbose=0)
            elif isRandom:
                loss, acc = model.evaluate([randomlist_ts, tf_ts], Y_ts, verbose=0)
            elif tfidf:
                loss, acc = model.evaluate([X_ts, tf_ts], Y_ts, verbose=0)
            else:
                loss, acc = model.evaluate([X_ts, bert_ts], Y_ts, verbose=0)
            print(str(i), ": loss = ", loss, " accuracy=", acc)
    if isRandom and not tfidf:
        scores = model.evaluate([randomlist_ts, bert_ts], Y_ts)
    elif isRandom:
        scores = model.evaluate([randomlist_ts, tf_ts], Y_ts)
    elif tfidf:
        scores = model.evaluate([X_ts, tf_ts], Y_ts)
    else:
        scores = model.evaluate([X_ts, bert_ts], Y_ts)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    if isRandom and not tfidf:
        return scores, model, randomlist_ts , bert_ts
    elif isRandom:
        return scores, model, randomlist_ts , tf_ts
    elif tfidf:
        return scores, model, X_ts , tf_ts
    else:
        return scores, model, X_ts , bert_ts


def preprocess_data(path):
    cvscores = []
    if three:
        models = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        all_l = []
        all_pred = []
    else:
        models = [[0, 0], [0, 0]]
    for i in range(0, 5):
        print(f'----------------------------------{i}----------------------------------')
        train_df = pd.read_csv(f'{path}/last_train{i}.csv', delimiter=',',encoding="ISO-8859-8")
        test_df = pd.read_csv(f'{path}/last_test{i}.csv', delimiter=',',encoding="ISO-8859-8")
        print(train_df)
        print(test_df)
        if three:
            train_df['label'].replace({'zero': 0, 'two':1, 'four': 2}, inplace=True)
            test_df['label'].replace({'zero': 0, 'two':1, 'four': 2}, inplace=True)

        else:
            if second:
                train_df['label'].replace({'zero': 0, 'four': 1}, inplace=True)
                test_df['label'].replace({'zero': 0, 'four': 1}, inplace=True)
            else:
                train_df['label'].replace({'zero': 0, 'two': 1}, inplace=True)
                test_df['label'].replace({'zero': 0, 'two': 1}, inplace=True)
        vecs_tr = []
        bert_tr = []
        vecs_ts = []
        bert_ts = []
        for row_ts in test_df.iterrows():
                # Convert the vector to list (from string representation)
                row_ts[1]['vec'] = row_ts[1]['vec'].replace('[', '')
                row_ts[1]['vec'] = row_ts[1]['vec'].replace(']', '')
                temp_ts = np.array(row_ts[1]['vec'].split(), dtype=np.float)

                vecs_ts.append(temp_ts)
                # Append the bag of words vector
                col = 'bert'
                row_ts[1][col] = row_ts[1][col].replace('[', '')
                row_ts[1][col] = row_ts[1][col].replace(']', '')
                b_ts = np.array(row_ts[1][col].split(), dtype=np.float)
                bert_ts.append(b_ts)
        test_df['vec'] = vecs_ts
        test_df['bert'] = bert_ts
        for row_tr in train_df.iterrows():
            # Convert the vector to list (from string representation)
            row_tr[1]['vec'] = row_tr[1]['vec'].replace('[', '')
            row_tr[1]['vec'] = row_tr[1]['vec'].replace(']', '')
            row_tr[1]['vec'] = row_tr[1]['vec'].replace('\n', '')
            row_tr[1]['vec'] = row_tr[1]['vec'].replace(')', '')
            row_tr[1]['vec'] = row_tr[1]['vec'].replace('array(', '')
            temp_tr = np.array(row_tr[1]['vec'].split(), dtype=np.float)
            vecs_tr.append(temp_tr)

            # Append the bag of words vector
            col = 'bert'
            row_tr[1][col] = row_tr[1][col].replace('[', '')
            row_tr[1][col] = row_tr[1][col].replace(']', '')
            b_tr = np.array(row_tr[1][col].split(), dtype=np.float)
            bert_tr.append(b_tr)
        train_df['vec'] = vecs_tr
        train_df['bert'] = bert_tr
        print(train_df)
        print(test_df)

        scores, modell , X_ts, bert_ts = train_and_predict(train_df, test_df)
        cvscores.append(scores[1] * 100)
        Y_ts = np.asarray([item for item in test_df.label])
        models += get_results(modell, X_ts, Y_ts, bert_ts)

        if three:
            y_pred = modell.predict([X_ts, bert_ts])
            if i == 0:
                all_l = Y_ts
                all_pred = np.argmax(y_pred, axis=1)
            else:
                all_l = [*all_l, *Y_ts]
                all_pred = [*all_pred, *np.argmax(y_pred, axis=1)]
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
    preprocess_data(path)
