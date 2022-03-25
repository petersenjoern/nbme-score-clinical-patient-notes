#%%
# Libraries
import pathlib
import yaml
from utils.misc import seed_env, load_and_prepare_nbme_data, format_annos_from_same_patient
import tensorflow as tf


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

# reformat texts and its annotations per patient
# to [{
#   "content": n_history, 
#   "annotations": [{
#       "start": location[0],
#       "stop": location[1],
#       "text": annotation,
#       "tag": feature_text
#   },{...}]
# },{...}]


#%%
data, unique_labels = format_annos_from_same_patient(df)
print(data, unique_labels)
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
