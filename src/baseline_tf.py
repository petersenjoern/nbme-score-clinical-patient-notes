#%%
## Regular imports
import os
import ast
import pathlib
import random
import numpy as np
import pandas as pd
import tensorflow as tf
print('TF version',tf.__version__)

import datasets
from datasets import load_dataset, Dataset
print(f"datasets version: {datasets.__version__}")

from tqdm.auto import tqdm
from typing import List, Dict
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

#%%
## Directories/ Paths
BASE_DIR = pathlib.Path(__file__).parents[1]
BASE_PATH_INPUT = pathlib.Path(__file__).parents[1].joinpath("data", "input")
OUTPUT_DIR = pathlib.Path(__file__).parents[1].joinpath("data", "working")
OUTPUT_DIR_TOKENIZER=OUTPUT_DIR.joinpath('tokenizer')


#%%
## Custom imports mostly due to root permissions in container
os.system('pip uninstall -y transformers')
os.system(f'python -m pip install --no-index --find-links={str(BASE_DIR.joinpath("data","input","nbme-pip-wheels-tokenizers"))} tokenizers')
import tokenizers
print(f"tokenizers.__version__: {tokenizers.__version__}")

os.system(f'python -m pip install --no-index --find-links={str(BASE_DIR.joinpath("data","input","nbme-pip-wheels-transformers"))} transformers')
import transformers
print(f"transformers.__version__: {transformers.__version__}")
from transformers import (AutoTokenizer, PreTrainedTokenizer, TFAutoModelForTokenClassification, AutoConfig,
    DataCollatorWithPadding, DataCollatorForTokenClassification, create_optimizer)

#%%
## Configuration & logging
class CFG:
    debug=True
    fold_n=5
    model="microsoft/deberta-base"
    epochs=5
    seed=42
    train=True
    tokenizer=None
    max_len=None
    batch_size=2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_logger(filename: str):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = get_logger(filename=__name__)
                                    
# %%
## Functions & Abstractions that are dataset/model specific
def seed_everything(seed:int=42) -> None:
    """Set seed of services for reproducibility."""

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["TF_CUDNN_DETERMINISTIC"] = str(seed)


def define_gpu_strategy():
    """Define the GPU strategy for modelling."""

    gpu_available = tf.config.list_physical_devices('GPU')

    if os.environ["CUDA_VISIBLE_DEVICES"].count(',') == 0:
        strategy = tf.distribute.get_strategy()
    else:
        strategy = tf.distribute.MirroredStrategy()
    
    return strategy

def set_mixed_precision():
    """Set TF mixed precision."""
    
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
    LOGGER.info("Set TF config to auto mixed precision=True")


def load_train_or_val_data(path_to_file: pathlib.Path) -> pd.DataFrame:
    """Load dataset for training or validation"""

    df = pd.read_csv(path_to_file)
    return _preprocess_train_or_val_data(df)

def _preprocess_train_or_val_data(df_train_or_val: pd.DataFrame) -> pd.DataFrame:
    """Preprocessing activities on the train or val df"""

    df_train_or_val['annotation'] = df_train_or_val['annotation'].apply(ast.literal_eval)
    df_train_or_val['location'] = df_train_or_val['location'].apply(ast.literal_eval)
    df_train_or_val['annotation_length'] = df_train_or_val['annotation'].apply(len)
    return df_train_or_val

def load_patient_notes(path_to_file: pathlib.Path) -> pd.DataFrame:
    """Load dataset of patient notes"""

    df = pd.read_csv(path_to_file)
    return df

def load_features(path_to_file: pathlib.Path) -> pd.DataFrame:
    """Load dataset of features"""

    df = pd.read_csv(path_to_file)
    return _preprocess_features(df)

def _preprocess_features(df_features: pd.DataFrame) -> pd.DataFrame:
    """preprocess logic for features"""
    
    df_features.loc[27, 'feature_text'] = "Last-Pap-smear-1-year-ago"
    return df_features

def get_tokenizer(modelname: str) -> PreTrainedTokenizer:
    """Initate, save and return tokenizer for modelname"""

    tokenizer = AutoTokenizer.from_pretrained(modelname)
    tokenizer.save_pretrained(OUTPUT_DIR_TOKENIZER)
    return tokenizer

def get_max_len_for_feature(df: pd.DataFrame, feature_names: List[str], tokenizer) -> int:
    """Calculate and return max len for feature in df"""

    for text_col in feature_names:
        if not isinstance(df[text_col][0], str):
            raise Exception("The feature is not a str")
        feature_lengths = []
        tk0 = tqdm(df[text_col].fillna("").values, total=len(df))
        for text in tk0:
            length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
            feature_lengths.append(length)
        LOGGER.info(f'{text_col} max(lengths): {max(feature_lengths)}')
    return max(feature_lengths)

def prepare_input(example: datasets.arrow_dataset.Example) -> Dict[str, int]:
    """
    Tokenize input and return the tokenization result.
    Returns: 'attention_mask', 'input_ids', 'token_type_ids'
    """

    inputs = CFG.tokenizer(example["pn_history"], example["feature_text"], 
                           add_special_tokens=True,
                           max_length=CFG.max_len,
                           padding="max_length",
                           return_offsets_mapping=False)
    return inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]


def prepare_label(example: datasets.arrow_dataset.Example) -> Dict[str, int]:
    encoded = CFG.tokenizer(example["pn_history"],
                            add_special_tokens=True,
                            max_length=CFG.max_len,
                            padding="max_length",
                            return_offsets_mapping=True)
    
    # The offsets_mapping is the tuple(start_char, stop_char) for token in the tokenizer input (pn_history here)
    # Hence, word != token; there may be multiple tokens per word, therefore the label idx has to be adjusted to the tokens
    offset_mapping = encoded['offset_mapping']
    # sequence_ids() are the ids (0's) where there is a real token; hence ignore all non-real tokens such as CLS, SEP, PAD etc.
    ignore_idxes = np.where(np.array(encoded.sequence_ids()) != 0)[0] 
    label = np.zeros(len(offset_mapping))
    label[ignore_idxes] = -100 # -100 for tensorflow; so they can be ignored during training
    if example["annotation_length"] != 0:
        # "location", this is the NER label idx with start_char, stop_char - can be multiple
        # preprocess the NER label indexes; eg. split them
        for location in example["location"]:
            for loc in [s.split() for s in location.split(';')]:
                start_idx = -1
                end_idx = -1
                start, end = int(loc[0]), int(loc[1])
                # align the label(s) to the tokens
                for idx in range(len(offset_mapping)):
                    if (start_idx == -1) & (start < offset_mapping[idx][0]):
                        start_idx = idx - 1
                    if (end_idx == -1) & (end <= offset_mapping[idx][1]):
                        end_idx = idx + 1
                if start_idx == -1:
                    start_idx = end_idx
                if (start_idx != -1) & (end_idx != -1):
                    label[start_idx:end_idx] = 1
    return label

def preprocess_input_and_label(example: datasets.arrow_dataset.Example) -> datasets.arrow_dataset.Example:
    """Preprocess input and labels at the same time"""

    example["input_ids"], example["attention_mask"], example["token_type_ids"] = prepare_input(example)
    example["labels"] = prepare_label(example)
    return example

def add_group_folds(df: pd.DataFrame, n_folds:int, group_col: str = "pn_num") -> pd.DataFrame:
    """Add column fold with group folds to dataframe based on groupby column"""

    df_copy = df.copy(deep=True)
    group_k_fold = GroupKFold(n_splits=n_folds)
    groups = df[group_col].values

    ## x and y in GroupKFold.split() are not essential; we just want the indexes
    for n, (_, idx) in enumerate(group_k_fold.split(df, df, groups)):
        df_copy.loc[idx, 'fold'] = int(n)
    df_copy['fold'] = df_copy['fold'].astype(int)
    group_sizes = df_copy.groupby('fold').size()
    LOGGER.info(f'Fold sizes in the dataset: {group_sizes}')

    return df_copy

#%%
## Execution (main function)

if __name__ == "__main__":
    seed_everything(42)
    
    # Load data
    df_train = load_train_or_val_data(BASE_PATH_INPUT.joinpath('train.csv'))
    df_patient_notes = load_patient_notes(BASE_PATH_INPUT.joinpath('patient_notes.csv'))
    df_features = load_features(BASE_PATH_INPUT.joinpath('features.csv'))

    df_train = df_train.merge(df_features, on=['feature_num', 'case_num'], how='left')
    df_train = df_train.merge(df_patient_notes, on=['pn_num', 'case_num'], how='left')

    # Add kfold to data; for cross validation in trainings loop
    df_train = add_group_folds(df_train, n_folds=CFG.fold_n, group_col="pn_num")


    # Tokenizer, Model
    CFG.tokenizer = get_tokenizer(CFG.model)
    max_len_pn_history = get_max_len_for_feature(df_train, ["pn_history"], CFG.tokenizer)
    max_len_feature_text = get_max_len_for_feature(df_train, ["feature_text"], CFG.tokenizer)
    CFG.max_len = max_len_pn_history + max_len_feature_text + len(["CLS", "SEP", "UNK"])
    #CLS sentence start, #SEP sentence end and new start, #UNK tokens not appearing in the original vocabulary,
    model_config = AutoConfig.from_pretrained(CFG.model, output_hidden_states=True)
    model = TFAutoModelForTokenClassification.from_pretrained(CFG.model)


    # Split train and test data
    dataset = Dataset.from_pandas(df_train)
    dataset_splitted=dataset.train_test_split(test_size=0.1)

    # Prepare test data
    dataset_transformed_test =  dataset_splitted["test"].map(
            preprocess_input_and_label,
            num_proc=os.cpu_count()-2
    )
    dataset_tf_test = dataset_transformed_test.to_tf_dataset(
        columns=["input_ids", "token_type_ids", "attention_mask"],
        label_cols=["labels"],
        batch_size=CFG.batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer=CFG.tokenizer, return_tensors="tf"),
        shuffle=True
    )

    # Prepare train and val data for each fold
    dataset_transformed = dataset_splitted["train"].map(
        preprocess_input_and_label,
        num_proc=os.cpu_count()-2,
    )
    
    iteration_n = 0
    dataset_transformed_train_fold = dataset_transformed.filter(lambda example: example['fold'] != iteration_n)
    dataset_transformed_val_fold = dataset_transformed.filter(lambda example: example['fold'] == iteration_n)

    dataset_tf_train = dataset_transformed_train_fold.to_tf_dataset(
        columns=["input_ids", "token_type_ids", "attention_mask"],
        label_cols=["labels"],
        batch_size=CFG.batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer=CFG.tokenizer, return_tensors="tf", max_length=CFG.max_len),
        shuffle=True
    )

    dataset_tf_val = dataset_transformed_val_fold.to_tf_dataset(
        columns=["input_ids", "token_type_ids", "attention_mask"],
        label_cols=["labels"],
        batch_size=CFG.batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer=CFG.tokenizer, return_tensors="tf", max_length=CFG.max_len),
        shuffle=True
    )

    num_train_steps = (dataset_transformed_train_fold.num_rows // CFG.batch_size) * CFG.epochs
    LOGGER.info(f"Number of training steps: {num_train_steps}")

    ## Create AdamWeightDecay and lr PolynomialDecay
    optimizer, lr_schedule = create_optimizer(
        init_lr=2e-5,
        num_train_steps=num_train_steps,
        weight_decay_rate=0.01,
        num_warmup_steps=0,
    )
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    LOGGER.info(model.summary())
    model.fit(x=dataset_tf_train, validation_data=dataset_tf_val, epochs=CFG.epochs)

#%%


# CFG.tokenizer.decode[ids here]
#%%

# %%
