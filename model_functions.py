"""
All functions related to wav2vec models

In this file you can define your requested models and labels.
"""

import os
import wandb
from shutil import copyfile

import librosa
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import torchaudio
from transformers import AutoConfig ,AutoModelForAudioClassification

from wav2vec2.model_classes import *

# Choose the requested model (from ModelName class).
MODEL = ModelName.BASE

# Define labels
# 0 = neutral speech
# 2 = insulting speech
# 4 = unsafe speech

# if both False = (0,2) classes
second = False  # Two classes (0,4)
three = True  # Three classes (0,2,4)


if three:
    label_list = ['zero', 'two', 'four']
else:
    label_list = ['zero', 'four']

pred_final = []
labels_final = []

# We need to specify the input and output column
input_column = "path"
output_column = "label"

save_path = "data"

if MODEL is ModelName.BASE:
    # Wav2Vec base - path
    model_name_or_path = "facebook/wav2vec2-base"

else: # emotion model
    # Wav2Vec with fine-tuning on emotion task - path
    model_name_or_path = 'ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition'

print(model_name_or_path)
# # DATA files paths
df_path = 'data/without-middle-0and4.csv'
root_path = "C:/Users/noaai/Desktop/merav_data/"


feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path, )
target_sampling_rate = feature_extractor.sampling_rate


def split_data_to_folders():
    """ split audio files by classes """
    counterT = 0
    counterF = 0

    df = pd.read_csv(df_path, encoding="ISO-8859-8")
    df = df.dropna(how='all')

    names = df["Type and content"]
    types = df.mark
    file_names = []
    for audio_filename, type in zip(names, types):
        if type == "0" or type == 0:
            counterT += 1
            name = "0/0_" + str(counterT) + ".wav"
            copyfile(root_path + audio_filename + ".wav", name)
            file_names.append(name)
        elif type == "4" or type == 4:
            counterF += 1
            name = "4/4_" + str(counterF) + ".wav"
            copyfile(root_path + audio_filename + ".wav", name)
            file_names.append(name)
        else:
            t = audio_filename
            audio_filename = type

            if t == "0" or t == 0:
                counterT += 1
                name = "0/0_" + str(counterT) + ".wav"
                copyfile(root_path + audio_filename + ".wav", name)
                file_names.append(name)
            elif t == "4" or t == 4:
                counterF += 1
                name = "4/4_" + str(counterF) + ".wav"
                copyfile(root_path + audio_filename + ".wav", name)
                file_names.append(name)

    df['file_name'] = file_names
    df.to_csv("data/without-middle-all-names.csv", encoding="ISO-8859-8")
    print("0:", counterT)
    print("4:", counterF)
    print("Sum:", (counterT + counterF))


def read_harmful_data():
    # list the files
    filelist = os.listdir('0')
    # read them into pandas
    df_zero = pd.DataFrame(filelist)
    # Adding the 1 label to the dataframe representing male
    df_zero['label'] = 'zero'
    # Renaming the column name to file
    df_zero = df_zero.rename(columns={0: 'path'})

    # Checking for a file that gets automatically generated and we need to drop
    df_zero[df_zero['path'] == '.DS_Store']
    df_zero['path'] = "0/" + df_zero['path']
    filelist = os.listdir('4')
    # read them into pandas
    df_four = pd.DataFrame(filelist)
    df_four['label'] = 'four'

    df_four = df_four.rename(columns={0: 'path'})

    df_four[df_four['path'] == '.DS_Store']
    df_four['path'] = "4/" + df_four['path']
    df = pd.concat([df_four, df_zero], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


def create_df_for_fold(df, train_idx, test_idx):
    """ Create df of the current fold each fold to train test according to the indexes"""

    save_path = "data"

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df.to_csv(f"{save_path}/train.csv", sep="\t", encoding="utf-8", index=False)
    test_df.to_csv(f"{save_path}/test.csv", sep="\t", encoding="utf-8", index=False)

    return train_df, test_df


def prepare_data(num_labels, label_list):
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        label2id={label: i for i, label in enumerate(label_list)},
        id2label={i: label for i, label in enumerate(label_list)},
        finetuning_task="wav2vec2_clf",
    )
    setattr(config, 'pooling_mode', "mean")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
    target_sampling_rate = feature_extractor.sampling_rate

    return config, feature_extractor, target_sampling_rate


def speech_file_to_array_fn(path):
    path = os.path.join(str(path))
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy().flatten()
    return speech


def label_to_id(label, label_list):
    if len(label_list) > 0:
        return label_list.index(label) if label in label_list else -1

    return label


def preprocess_function(examples):
    speech_list = [speech_file_to_array_fn(path) for path in examples[input_column]]
    target_list = [label_to_id(label, label_list) for label in examples[output_column]]
    result = feature_extractor(speech_list, sampling_rate=target_sampling_rate)
    result["labels"] = list(target_list)
    return result


def compute_metrics(pred):
    if three:
        label_idx = [0, 1, 2]
    else:
        label_idx = [0, 1]
    label_names = label_list

    preds = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    preds = np.argmax(preds, axis=1)
    labels = pred.label_ids
    acc = (preds == pred.label_ids).astype(np.float32).mean().item()
    f1 = f1_score(labels, preds, average='macro')
    pred_final.append(preds)
    labels_final.append(labels)

    report = classification_report(y_true=labels, y_pred=preds, labels=label_idx, target_names=label_names)
    matrix = confusion_matrix(y_true=labels, y_pred=preds)
    print(report)
    print(matrix)

    wandb.log(
        {"conf_mat": wandb.plot.confusion_matrix(probs=None, y_true=labels, preds=preds, class_names=label_names)})

    return {"accuracy": acc, "f1_score": f1}


def speech_file_to_array_fn2(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    speech_array = speech_array.squeeze().numpy().flatten()
    speech_array = librosa.resample(np.asarray(speech_array), sampling_rate, feature_extractor.sampling_rate)
    batch["speech"] = speech_array
    return batch


def get_vec_from_pre_trained(batch):
    """
       Get the audio vector embedding from the Wav2Vec model - no fine tuning.
       :param batch: batch of audio samples
       :return: audio embedding vectors of the given batch
       """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = AutoConfig.from_pretrained(model_name_or_path)
    setattr(config, 'pooling_mode', "mean")
    model = Wav2Vec2ForSpeechClassificationNoFineTune.from_pretrained(
        model_name_or_path,
        config=config,

    ).to(device)

    features = feature_extractor(batch["speech"], sampling_rate=feature_extractor.sampling_rate,
                                 return_tensors="pt", padding=True)
    input_values = features.input_values.to(device)
    with torch.no_grad():
        res = model(input_values)
        res = res.tolist()
    batch["vec"] = res
    print(res)  # print the audio embedding vectors

    return batch


def get_vec_of_8(batch):
    """
       Get the audio vector of 8 emotion from the emotion fine-tuned model - no fine tuning on our data.
       :param batch: batch of audio samples
       :return: audio embedding vectors of the given batch
       """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
    model = AutoModelForAudioClassification.from_pretrained(
        model_name_or_path,
    ).to(device)

    features = feature_extractor(batch["speech"], sampling_rate=feature_extractor.sampling_rate,
                                 return_tensors="pt", padding=True)
    input_values = features.input_values.to(device)
    with torch.no_grad():
        # take the logits of the 8 emotions
        res = model(input_values).logits
        res = res.tolist()
        print(res)
        batch["vec"] = res
    return batch


def get_vec_from_trained_model(batch):
    """
        Get the audio vector embedding from the Wav2Vec model - ** after ** fine-tuning.
        :param batch: batch of audio samples
        :return: audio embedding vectors of the given batch
        """
    model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to('cuda')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = feature_extractor(batch["speech"], sampling_rate=feature_extractor.sampling_rate,
                                 return_tensors="pt", padding=True)
    input_values = features.input_values.to(device)
    print(input_values)
    with torch.no_grad():
        res = model(input_values).hidden_states
        res = res.tolist()
    batch["vec"] = res
    return batch
