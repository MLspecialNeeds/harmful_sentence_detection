"""
- Extract the audio embedding vector from wav2vec pre-trained model.
-Save results in a csv file (data/{model_name}_vecs.csv).

* First define reuqested model in 'model_functions' file.
* For embedding after fine-tune , fine tune use 'run_wav2vec' file (the embedding is in the 'Prediction' section).
"""

from datasets import load_dataset
from pandas import read_csv

import torch

torch.cuda.empty_cache()
from model_functions import *

# path to the df of the audio samples
df_path = "data/merav/merav2-all.csv"

def main():
    df = read_csv(df_path)
    print(df.head())

    df.to_csv(f"{save_path}/all.csv", sep="\t", encoding="utf-8", index=False)

    num_labels = len(label_list)
    print(f"A classification problem with {num_labels} classes: {label_list}")

    config, _, _ = prepare_data(num_labels, label_list)
    print(f"The target sampling rate: {target_sampling_rate}")

    df = load_dataset("csv", data_files={"all_df": "data/all.csv"}, delimiter="\t")["all_df"]
    df = df.map(
        speech_file_to_array_fn2
    )
    # Apply the extraction function
    if MODEL is ModelName.EIGHT_EMOTIONS:
        result = df.map(get_vec_of_8, batch_size=10,batched=True,num_proc=1)
    else:
        result = df.map(get_vec_from_pre_trained, batched=True, batch_size=2) #without finetuning

    # Save to vectors to csv file
    result.to_csv(f"{save_path}/{MODEL}_vecs.csv", sep=",", encoding="utf-8", index=False)

if __name__ == "__main__":
    main()
