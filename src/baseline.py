#%%
# Libraries
import pathlib
import yaml
from utils.misc import seed_env, load_and_prepare_nbme_data
import tensorflow as tf
import pandas as pd
from typing import Tuple, List

# Paths
PATH_BASE = pathlib.Path(__file__).absolute().parents[1]
PATH_YAML = PATH_BASE.joinpath("config.yaml")

# Constants
EPSILON = tf.keras.backend.epsilon() #to prevent divide by zero error


# Setup
with open(PATH_YAML, "r") as file:
    cfg = yaml.safe_load(file)

seed_env(424)


#%%
cfg.get("datasets")
df=load_and_prepare_nbme_data(paths=cfg["datasets"], train=True)


# %%
df.head()


#%%
# convert annotation, location, feature_text, pn_history
# to [{"content": n_history, "annotations": [{"start": location[0], "stop": location[1], "text": annotation, "tag": feature_text}]}]

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


#TODO: replace df.loc[df["location"] == "['85 99', '126 138', '126 131;143 151']"] 
def extract_all_from_same_content(df: pd.DataFrame) -> Tuple[List, List]:
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
            
            # clean "['696 724', '123 456']" and "['intermittent episodes', 'episode']"
            anno_locs = row["location"].strip("']['").split(', ')
            annos = row["annotation"].replace("'","").strip("][").split(', ')
            for anno_loc, anno in zip(anno_locs, annos):
                start, end = tuple(anno_loc.replace("'", "").split(" "))
                start, end = int(start), int(end)
                text = content[start:end]
                assert text == anno
                annotations.append(dict(start=start, end=end, text=text, tag=tag))
            preprocessed_example = {"content": content, "annotations": annotations}
            preprocessed_data.append(preprocessed_example)
    return preprocessed_data, unique_labels

#%%
oo, rr = extract_all_from_same_content(df)

# %%
# from tf.keras.utils import Sequence
# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")


# %%


# class CustomDataset(Sequence):
#     def __init__(
#         self,
#         data,
#         label_set,
#         tokenizer,
#         tokens_per_batch,
#         window_stride,
#     ):
#     self.text = []
#     self.annotations = []
#     self.label_set = []
