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
from transformers import AutoTokenizer, AutoModel, AutoConfig, DataCollatorWithPadding

#%%
## Configuration
class CFG:
    debug=True
    fold_n=5
    modelName="microsoft/deberta-base"
    epochs=5
    seed=42
    train=True
    tokenizer=None
    max_len=None
    batch_size=2


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

                                    
# %%
## Functions & Abstractions that are dataset/model specific
def seed_everything(seed:int=42) -> None:
    """Set seed of services for reproducibility."""

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["TF_CUDNN_DETERMINISTIC"] = str(seed)

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

#TODO: define return type
def get_tokenizer(modelname: str):
    """Initate, save and return tokenizer for modelname"""

    tokenizer = AutoTokenizer.from_pretrained(modelname)
    tokenizer.save_pretrained(OUTPUT_DIR_TOKENIZER)
    return tokenizer

#TODO: add logging instead of print
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
        print(f'{text_col} max(lengths): {max(feature_lengths)}')
    return max(feature_lengths)

# TODO: example type annotation (dataset one record)
def prepare_input(example) -> Dict[str, int]:
    """Tokenize input and return the tokenization result.
    Returns: 'attention_mask', 'input_ids', 'token_type_ids'
    """

    inputs = CFG.tokenizer(example["pn_history"], example["feature_text"], 
                           add_special_tokens=True,
                           max_length=CFG.max_len,
                           padding="max_length",
                           return_offsets_mapping=False)
    return inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]


# TODO: step through and annotate and comment
def prepare_label(example) -> Dict[str, int]:
    encoded = CFG.tokenizer(example["pn_history"],
                            add_special_tokens=True,
                            max_length=CFG.max_len,
                            padding="max_length",
                            return_offsets_mapping=True)
    offset_mapping = encoded['offset_mapping']
    ignore_idxes = np.where(np.array(encoded.sequence_ids()) != 0)[0]
    label = np.zeros(len(offset_mapping))
    label[ignore_idxes] = -1
    if example["annotation_length"] != 0:
        for location in example["location"]:
            for loc in [s.split() for s in location.split(';')]:
                start_idx = -1
                end_idx = -1
                start, end = int(loc[0]), int(loc[1])
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

def preprocess_input_and_label(example):
    """Preprocess input and labels at the same time"""
    example["input_ids"], example["attention_mask"], example["token_type_ids"] = prepare_input(example)
    example["labels"] = prepare_label(example)
    return example

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


    # Add KFold label to data
    #TODO: extract this as a function
    Fold = GroupKFold(n_splits=CFG.fold_n)
    groups = df_train['pn_num'].values
    for n, (_, val_index) in enumerate(Fold.split(df_train, df_train['location'], groups)):
        df_train.loc[val_index, 'fold'] = int(n)
    df_train['fold'] = df_train['fold'].astype(int)
    print(df_train.groupby('fold').size())
    
    # Tokenizer, Model
    CFG.tokenizer = get_tokenizer(CFG.modelName)
    max_len_pn_history = get_max_len_for_feature(df_train, ["pn_history"], CFG.tokenizer)
    max_len_feature_text = get_max_len_for_feature(df_train, ["feature_text"], CFG.tokenizer)
    CFG.max_len = max_len_pn_history + max_len_feature_text + len(["CLS", "SEP", "SEP"])

    # Prepare train and test data
    dataset = Dataset.from_pandas(df_train)
    dataset_splitted=dataset.train_test_split(test_size=0.1)


    # TODO: preprocess test data as well dataset_splitted["test"]
    dataset_tf_test = dataset_splitted_xxxxx["test"].to_tf_dataset(
        columns=["input_ids", "token_type_ids", "attention_mask"],
        label_cols=["labels"],
        batch_size=CFG.batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer=CFG.tokenizer, return_tensors="tf"),
        shuffle=True
    )



    # Prepare data for each fold
    REMOVE_COLS = ['id', 'case_num', 'pn_num', 'feature_num', 'annotation', 'location', 'annotation_length', 'feature_text', 'pn_history', '__index_level_0__']
    dataset_transformed = dataset["train"].map(preprocess_input_and_label, num_proc=os.cpu_count()-2, remove_columns=REMOVE_COLS)
    
    iteration_n = 0
    dataset_transformed_train_fold = dataset_transformed.filter(lambda example: example['fold'] != iteration_n)
    dataset_transformed_val_fold = dataset_transformed.filter(lambda example: example['fold'] == iteration_n)

    dataset_tf_train = dataset_transformed_train_fold.to_tf_dataset(
        columns=["input_ids", "token_type_ids", "attention_mask"],
        label_cols=["labels"],
        batch_size=CFG.batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer=CFG.tokenizer, return_tensors="tf"),
        shuffle=True
    )

    dataset_tf_val = dataset_transformed_val_fold.to_tf_dataset(
        columns=["input_ids", "token_type_ids", "attention_mask"],
        label_cols=["labels"],
        batch_size=CFG.batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer=CFG.tokenizer, return_tensors="tf"),
        shuffle=True
    )


#%%



#%%

# %%
