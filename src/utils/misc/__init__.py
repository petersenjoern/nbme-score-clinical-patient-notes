import os
import re
import random
import numpy as np
import tensorflow as tf
import pandas as pd
from typing import Dict

def seed_env(seed: int) -> None:
    """seed various services & libraries"""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_and_prepare_nbme_data(paths: Dict[str, str], train:bool=False) -> pd.DataFrame:
    """Loading data files and merge"""

    if train:
        df = pd.read_csv(paths.get("train"))
    else:
        df = pd.read_csv(paths.get("test"))
    
    features = pd.read_csv(paths.get("features"))
    patient_notes = pd.read_csv(paths.get("patient_notes"))

    df = df.merge(features, how="left", on=["case_num", "feature_num"])
    df = df.merge(patient_notes, how="left", on=["case_num", "pn_num"])


    df["pn_history"] = df["pn_history"].apply(lambda x: x.strip())
    df["pn_history"] = df["pn_history"].apply(normalize_spaces)
    
    df["feature_text"] = df["feature_text"].apply(normalize_feature_text)
    df["feature_text"] = df["feature_text"].apply(normalize_spaces)

    df = manual_curation_of_entries(df)

    return df


def normalize_feature_text(feature_text: str) -> str:
    """various normalization of the 144 features"""

    feature_text = re.sub('I-year', '1-year', feature_text)
    feature_text = re.sub('-OR-', " or ", feature_text)
    feature_text = re.sub('-', ' ', feature_text)
    return feature_text

def normalize_spaces(text: str) -> str:
    """normalize various breakpoints to ' '. """

    text = re.sub('\n', ' ', text)
    text = re.sub('\t', ' ', text)
    text = re.sub('\r', ' ', text)
    return text

def manual_curation_of_entries(df: pd.DataFrame) -> pd.DataFrame:
    """curate entries manually"""
    
    df.loc[(df["location"] == "['85 99', '126 138', '126 131;143 151']"), "location"] = "['85 99', '126 151']"
    return df
