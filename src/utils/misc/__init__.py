import os
import re
import random
import numpy as np
import tensorflow as tf
import pandas as pd
from typing import Dict, Tuple, List

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


def yield_rows_with_same_patient_id(df: pd.DataFrame):
    """
    Yield all records that have the same patient history so,
    they can be processed together subsequently
    """

    pn_nums = set()
    for _,row in df.iterrows():
        pn_num = int(row["pn_num"])
        if pn_num in pn_nums:
            continue

        pn_nums.add(pn_num)
        # for each patient return all valid (not empty []) rows with annotations
        mask = (df["pn_num"] == pn_num) & (df["annotation"] != "[]")
        df_for_one_patient = df.loc[mask]
        yield df_for_one_patient


def format_annos_from_same_patient(df: pd.DataFrame) -> Tuple[List, List]:
    """
    For each patient dataset (each contains 16 rows), prepare the data
    to a {"content": "patient has visited the doctor on the xx", 
    "annotations": [{"start": 1, "end": 2, "text": "at", "tag": "something"}]}
    """

    preprocessed_data = []
    unique_labels = set()
    for patient_subset_df in yield_rows_with_same_patient_id(df):
        annotations = []
        for _,row in patient_subset_df.iterrows():
            tag = str(row["feature_text"])
            unique_labels.add(tag)
            content = str(row["pn_history"])
            
            # clean annotation locations like:
            # "['696 724', '123 456']" and "['intermittent episodes', 'episode']"
            annos = row["annotation"].replace("'","").strip("][").split(', ') # can be used as assert
            anno_locs = row["location"].strip("']['").split(', ')
            # for each list item, try to split other signs like ;
            anno_locs = [anno_loc.replace("'", "").split(";") for anno_loc in anno_locs]
            
            #flatten list of lists to list
            anno_locs = [item for sublist in anno_locs for item in sublist]
            for anno_loc, _anno in zip(anno_locs, annos):
                start, end = tuple(anno_loc.replace("'", "").split(" "))
                start, end = int(start), int(end)
                text = content[start:end]
                annotations.append(dict(start=start, end=end, text=text, tag=tag))
            preprocessed_example = {"content": content, "annotations": annotations}
            preprocessed_data.append(preprocessed_example)
    return preprocessed_data, unique_labels
