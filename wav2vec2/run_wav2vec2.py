"""
This is the main file using for fine-tuning wav2vec models
****** First define model type and labels in model_function file ******
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['WANDB_LOG_MODEL'] = 'true'

from datasets import load_dataset, Dataset
from transformers import TrainingArguments

import sys
import numpy
import ast

from model_functions import *
from model_classes import *
import tensorflow as tf

numpy.set_printoptions(threshold=sys.maxsize)
torch.cuda.empty_cache()

# Define seed
tf.random.set_seed(42)
torch.manual_seed(42)


pred_final = []
labels_final = []


def main():
    split_data_to_folders()
    print(model_name_or_path)
    if second:
        df = pd.read_csv('../data/merav2-all.csv', delimiter=',')
    elif three:
        df = pd.read_csv('../data/merav3-all.csv', delimiter=',')
    else:  # 0 and 2 classes
        df = pd.read_csv('../data/merav_all.csv', delimiter=',')

    df = read_harmful_data(df)
    print(df.head())

    # Read fold indexes to be identical in all devices
    if second:
        with open('data/train2_idx.txt', 'r') as ftr:
            train_idx = ast.literal_eval(ftr.read())
        with open('data/test2_idx.txt', 'r') as fts:
            test_idx = ast.literal_eval(fts.read())
    elif three:
        with open('data/train3_idx.txt', 'r') as ftr:
            train_idx = ast.literal_eval(ftr.read())
        with open('data/test3_idx.txt', 'r') as fts:
            test_idx = ast.literal_eval(fts.read())
    else:
        with open('data/train_idx.txt', 'r') as ftr:
            train_idx = ast.literal_eval(ftr.read())
        with open('data/test_idx.txt', 'r') as fts:
            test_idx = ast.literal_eval(fts.read())

    # K-fold Cross Validation model
    fold = 0
    for train_ids, test_ids in zip(train_idx, test_idx):
        # ######################################## Prepare Data for Training #########################################
        print(f'FOLD {fold}')
        train_df, test_df = create_df_for_fold(df, train_ids, test_ids)
        print(train_df.shape)
        print(test_df.shape)

        data_files = {
            "train": "data/train.csv",
            "validation": "data/test.csv",
        }

        dataset = load_dataset("csv", data_files=data_files, delimiter="\t", )
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]

        print(train_dataset)
        print(eval_dataset)

        num_labels = len(label_list)
        print(f"A classification problem with {num_labels} classes: {label_list}")

        config, feature_extractor, target_sampling_rate = prepare_data(num_labels, label_list)
        print(f"The target sampling rate: {target_sampling_rate}")

        # ######################################## Preprocess Data #########################################

        train_dataset = train_dataset.map(
            preprocess_function,
            batch_size=10,
            batched=True,
            num_proc=1
        )
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batch_size=10,
            batched=True,
            num_proc=1
        )
        idx = 0
        print(f"Training labels: {train_dataset[idx]['labels']} - {train_dataset[idx]['label']}")

        # ######################################## Training (fine-tuning) #########################################

        data_collator = DataCollatorCTCWithPadding(feature_extractor=feature_extractor, padding=True)
        model = Wav2Vec2ForSpeechClassification.from_pretrained(
            model_name_or_path,
            config=config,
            tr_labels=train_dataset['label'],

        )
        model.freeze_feature_extractor()
        if fold >= 0:
            training_args = TrainingArguments(
                output_dir="",
                per_device_train_batch_size=3,
                per_device_eval_batch_size=2,
                gradient_accumulation_steps=2,
                evaluation_strategy="steps",
                num_train_epochs=10,  # 3.5
                load_best_model_at_end=True,
                fp16=True,
                save_steps=100,
                eval_steps=100,
                logging_steps=10,
                learning_rate=1e-5,
                save_total_limit=1,
                do_eval=True,
                do_train=True,
            )

            print(training_args)
            wandb.init(name=training_args.output_dir, config=training_args)
            wandb.init(project='my-updated-cheat-project', group='emotion_3_2_10_0.5_new_tanh_s_model',
                       reinit=True, name="f-" + str(fold))

            trainer = CTCTrainer(
                model=model,
                data_collator=data_collator,
                args=training_args,
                compute_metrics=compute_metrics,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=feature_extractor,
            )
            train_result = trainer.train()

            metrics = train_result.metrics

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()
            print(trainer)

            # Evaluation
            if training_args.do_eval:
                metrics = trainer.evaluate()

                trainer.log_metrics("eval", metrics)
                trainer.save_metrics("eval", metrics)

        # ######################################## Prediction #########################################
        if three:
            results = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        else:
            results = [[0, 0], [0, 0]]

        print("########## PREDICT TRAIN #############")
        metrics = trainer.predict(train_dataset)
        res = metrics.predictions[1]
        tr_df = pd.read_csv('data/train.csv')
        vectors = [item for item in res]
        print(*vectors)  # print all train embedding vectors to be used later
        tr_df["vec"] = vectors
        # Save df with the vectors embedding (if length greater than 1000 used the printed vectors)
        name = "last_train" + str(fold)
        save_path = "../data"
        tr_df.to_csv(f"{save_path}/{name}.csv", sep=",", encoding="utf-8", index=False)

        print("########## PREDICT TEST #############")
        metrics = trainer.predict(eval_dataset)
        res = metrics.predictions[1]
        ts_df = pd.read_csv('data/test.csv').copy()
        vectors = [item for item in res]

        print(*vectors) # print all test embedding vectors to be used later
        ts_df["vec"] = vectors
        # Save df with the vectors embedding (if length greater than 1000 used the printed vectors)
        name = "last_test" + str(fold)
        save_path = "../data"
        print(res)
        ts_df.to_csv(f"{save_path}/{name}.csv", sep=",", encoding="utf-8", index=False)

        # Create confusion matrix
        y_true = [label_to_id(name, label_list) for name in eval_dataset["label"]]
        y_pred = np.argmax(metrics.predictions[0], axis=1)
        if three:
            labels = [0, 1, 2]
        else:
            labels = [0, 1]
        label_names = [config.id2label[i] for i in range(config.num_labels)]
        results += confusion_matrix(y_true, y_pred, labels=labels)
        print(classification_report(y_true, y_pred, target_names=label_names))
        report = classification_report(y_true=labels_final, y_pred=pred_final, labels=labels,
                                       target_names=label_list)
        matrix = confusion_matrix(y_true=labels_final, y_pred=pred_final)
        print(report)
        print(matrix)

        wandb.log(
            {"prediction_mat": wandb.plot.confusion_matrix(probs=None, y_true=y_true, preds=y_pred,
                                                           class_names=label_names)})
        print(classification_report(y_true, y_pred, target_names=label_names))
        wandb.finish()

        fold = fold + 1

    wandb.log(
        {"conf_mat": wandb.plot.confusion_matrix(probs=None, y_true=labels_final, preds=pred_final,
                                                 class_names=label_list)})
    print(results)


if __name__ == "__main__":
    main()
