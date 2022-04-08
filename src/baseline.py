#%%
# Libraries
import pathlib
import yaml
from utils.misc import seed_env, load_and_prepare_nbme_data, format_annotations_from_same_patient, LabelSet
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from transformers import AutoTokenizer
from typing import Dict, Iterator, List, Tuple, Union, Any
from dataclasses import dataclass

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
data, unique_labels = format_annotations_from_same_patient(df)
print(data[0], unique_labels)

label_set = LabelSet(labels=unique_labels)
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# %%

@dataclass
class TrainingExample:
    input_ids: List[int]
    attention_masks: List[int]
    labels: List[int]
class CustomDataset(Sequence):
    def __init__(
        self,
        data,
        label_set,
        tokenizer,
        tokens_per_batch=32,
        window_stride=None,
        *args,
        **kwargs
    ):
        self.label_set = label_set
        if window_stride is None:
            self.window_stride = tokens_per_batch
        self.tokenizer = tokenizer
        for example in data:
            # changes tag key to label
            for a in example["annotations"]:
                a["label"] = a["tag"]
        self.texts = []
        self.annotations = []

        # Move up in loop above
        for example in data:
            self.texts.append(example["content"])
            self.annotations.append(example["annotations"])
        ###TOKENIZE All THE DATA
        tokenized_batch = self.tokenizer(self.texts, add_special_tokens=False)
        ###ALIGN LABELS ONE EXAMPLE AT A TIME
        aligned_labels = []
        for ix in range(len(tokenized_batch.encodings)):
            encoding = tokenized_batch.encodings[ix]
            raw_annotations = self.annotations[ix]
            aligned = label_set.get_aligned_label_ids_from_annotations(
                encoding, raw_annotations
            )
            aligned_labels.append(aligned)
        ###END OF LABEL ALIGNMENT

        ###MAKE A LIST OF TRAINING EXAMPLES. (This is where we add padding)
        self.training_examples: List[TrainingExample] = []
        for encoding, label in zip(tokenized_batch.encodings, aligned_labels):
            length = len(label)  # How long is this sequence
            for start in range(0, length, self.window_stride):

                end = min(start + tokens_per_batch, length)

                # How much padding do we need ?
                padding_to_add = max(0, tokens_per_batch - end + start)
                self.training_examples.append(
                    TrainingExample(
                        # Record the tokens
                        input_ids=encoding.ids[start:end]  # The ids of the tokens
                        + [self.tokenizer.pad_token_id]
                        * padding_to_add,  # padding if needed
                        labels=(
                            label[start:end]
                            + [-100] * padding_to_add  # padding if needed
                        ),  # -100 is a special token for padding of labels,
                        attention_masks=(
                            encoding.attention_mask[start:end]
                            + [0]
                            * padding_to_add  # 0'd attention masks where we added padding
                        ),
                    )
                )


    def __len__(self):
        """Return length of data processed"""
        return len(self.training_examples)

    def __getitem__(self, idx) -> TrainingExample:
        """Return one full preprocessed example by index"""
        return self.training_examples[idx]

#%%
x=CustomDataset(data=data, label_set=label_set, tokenizer=tokenizer)

# %%
x[0]
# %%
