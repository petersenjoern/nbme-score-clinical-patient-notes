#%%
# Libraries
import pathlib
import yaml
from utils.misc import seed_env, load_and_prepare_nbme_data
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


# %%
df.head(15)


#%%
# convert annotation, location, feature_text, pn_history
# to [{"content": n_history, "annotations": [{"start": location[0], "stop": location[1], "text": annotation, "tag": feature_text}]}]

def yield_rows_with_same_patient_id(df):
    """Yield all records that have the same patient history so,
    they can be processed together subsequently."""

    ##ensure here that there is actually an annotation,
    ## hence the annotation is not []
    contents = set()
    for _,row in df.iterrows():
        content = str(row["pn_num"])
        if content in contents:
            continue

        contents.add(content)
        mask = df["pn_history"] == content
        df_with_same_content = df.loc[mask]
        yield df_with_same_content

def extract_all_from_same_content(df):
    unique_labels = set()
    pass


# %%
from tf.keras.utils import Sequence
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")


# %%


class CustomDataset(Sequence):
    def __init__(
        self,
        data,
        label_set,
        tokenizer,
        tokens_per_batch,
        window_stride,
    ):
    self.text = []
    self.annotations = []
    self.label_set = []
