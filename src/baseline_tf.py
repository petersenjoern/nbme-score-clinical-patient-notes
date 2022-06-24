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

from tqdm.auto import tqdm
from typing import List


#%%
## Directories/ Paths
BASE_DIR = pathlib.Path(__file__).parents[1]
BASE_PATH_INPUT = pathlib.Path(__file__).parents[1].joinpath("data", "input")
OUTPUT_DIR = pathlib.Path(__file__).parents[1].joinpath("data", "working")
OUTPUT_DIR_TOKENIZER=OUTPUT_DIR.joinpath('tokenizer')


#%%
## Custom imports
os.system('pip uninstall -y transformers')
# pip download transformers && pip download tokenizers
os.system(f'python -m pip install --no-index --find-links={str(BASE_DIR.joinpath("data","input","nbme-pip-wheels-transformers"))} transformers')
os.system(f'python -m pip install --no-index --find-links={str(BASE_DIR.joinpath("data","input","nbme-pip-wheels-tokenizers"))} tokenizers')
import tokenizers
import transformers
print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig


#%%
## Configuration
class CFG:
    debug=True
    modelName="microsoft/deberta-base"
    epochs=5
    seed=42
    train=True

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

                                    
# %%
## Functions & Abstractions that are dataset/model specific
def seed_everything(seed:int=42) -> None:
    """Set seed of services for reproducibility."""

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

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
    
    # Tokenizer, Model
    tokenizer = get_tokenizer(CFG.modelName)
    max_len_pn_history = get_max_len_for_feature(df_train, ["pn_history"], tokenizer)
    max_len_feature_text = get_max_len_for_feature(df_train, ["feature_text"], tokenizer)
    max_len = max_len_pn_history + max_len_feature_text + len(["CLS", "SEP", "SEP"])


#%%


# %%
