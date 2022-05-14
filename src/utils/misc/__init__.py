import os
import re
import random
import numpy as np
import pandas as pd
import itertools
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from dataclasses import dataclass
from sklearn.metrics import classification_report
from typing import Dict, Iterator, List, Tuple, Union
from collections import defaultdict



def seed_env(seed: int) -> None:
    """seed various services & libraries"""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_and_prepare_nbme_data(paths: Dict[str, str], train: bool = False) -> pd.DataFrame:
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

    if train:
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

    df.loc[(df["location"] == "['85 99', '126 138', '126 131;143 151']"),
           "location"] = "['85 99', '126 151']"
    return df


def yield_rows_with_same_patient_id(df: pd.DataFrame):
    """
    Yield all records that have the same patient history so,
    they can be processed together subsequently
    """

    pn_nums = set()
    for _, row in df.iterrows():
        pn_num = int(row["pn_num"])
        if pn_num in pn_nums:
            continue

        pn_nums.add(pn_num)
        # for each patient return all valid (not empty []) rows with annotations
        mask = (df["pn_num"] == pn_num) & (df["annotation"] != "[]")
        df_for_one_patient = df.loc[mask]
        yield df_for_one_patient


def format_annotations_from_same_patient(df: pd.DataFrame) -> Tuple[List, List]:
    """
    For each patient dataset (each contains 16 rows), prepare the data
    to a {"content": "patient has visited the doctor on the xx", 
    "annotations": [{"start": 1, "end": 2, "text": "at", "tag": "something"}]}
    """

    preprocessed_data = []
    unique_labels = set()
    for patient_subset_df in yield_rows_with_same_patient_id(df):
        annotations = []
        for _, row in patient_subset_df.iterrows():
            tag = str(row["feature_text"])
            unique_labels.add(tag)
            content = str(row["pn_history"])

            # clean annotation locations like:
            # "['696 724', '123 456']" and "['intermittent episodes', 'episode']"
            annos = row["annotation"].replace("'", "").strip(
                "][").split(', ')  # can be used as assert
            anno_locs = row["location"].strip("']['").split(', ')
            # for each list item, try to split other signs like ;
            anno_locs = [anno_loc.replace("'", "").split(";")
                         for anno_loc in anno_locs]

            # flatten list of lists to list
            anno_locs = [item for sublist in anno_locs for item in sublist]
            for anno_loc, _anno in zip(anno_locs, annos):
                start, end = tuple(anno_loc.replace("'", "").split(" "))
                start, end = int(start), int(end)
                text = content[start:end]
                annotations.append(
                    dict(start=start, end=end, text=text, tag=tag))
            preprocessed_example = {
                "content": content, "annotations": annotations}
            preprocessed_data.append(preprocessed_example)
    return preprocessed_data, unique_labels


class LabelSet:
    """ Align target labels with tokens"""

    def __init__(self, labels: List[str]):
        """ Create BILU target labels"""

        self.labels_to_id = {}
        self.ids_to_label = {}
        self.labels_to_id["O"] = 0
        self.ids_to_label[0] = "O"
        num = 0  # in case there are no labels
        # Writing BILU will give us incremental ids for the labels
        for _num, (label, s) in enumerate(itertools.product(labels, "BILU")):
            num = _num + 1  # skip 0
            l = f"{s}-{label}"
            self.labels_to_id[l] = num
            self.ids_to_label[num] = l
        # Add the OUTSIDE label - no label for the token

    def __getitem__(self, item):
        return getattr(self, item)

    def align_tokens_and_annotations_bilou(self, tokenized, annotations):
        """align the tokens with the annotations in BILU"""

        tokens = tokenized.tokens
        aligned_labels = ["O"] * len(
            tokens
        )  # Make a list to store our labels the same length as our tokens
        for anno in annotations:
            annotation_token_ix_set = (
                set()
            )  # A set that stores the token indices of the annotation
            for char_ix in range(anno["start"], anno["end"]):

                token_ix = tokenized.char_to_token(char_ix)
                if token_ix is not None:
                    annotation_token_ix_set.add(token_ix)
            if len(annotation_token_ix_set) == 1:
                # If there is only one token
                token_ix = annotation_token_ix_set.pop()
                prefix = (
                    "U"  # This annotation spans one token so is prefixed with U for unique
                )
                aligned_labels[token_ix] = f"{prefix}-{anno['label']}"

            else:

                last_token_in_anno_ix = len(annotation_token_ix_set) - 1
                for num, token_ix in enumerate(sorted(annotation_token_ix_set)):
                    if num == 0:
                        prefix = "B"
                    elif num == last_token_in_anno_ix:
                        prefix = "L"  # Its the last token
                    else:
                        prefix = "I"  # We're inside of a multi token annotation
                    aligned_labels[token_ix] = f"{prefix}-{anno['tag']}"
        return aligned_labels

    def get_aligned_label_ids_from_annotations(self, tokenized_text, annotations) -> List:
        raw_labels = self.align_tokens_and_annotations_bilou(
            tokenized_text, annotations)
        return list(map(self.labels_to_id.get, raw_labels))



@dataclass
class TrainingExample:
    input_ids: List[int]
    attention_masks: List[int]
    labels: List[int]


class CustomDataset(Dataset):
    def __init__(
        self,
        data,
        label_set: LabelSet,
        tokenizer: AutoTokenizer,
        tokens_per_batch: int = 32,
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
        # TOKENIZE All THE DATA
        tokenized_batch = self.tokenizer(self.texts, add_special_tokens=False)
        # ALIGN LABELS ONE EXAMPLE AT A TIME
        aligned_labels = []
        for ix in range(len(tokenized_batch.encodings)):
            encoding = tokenized_batch.encodings[ix]
            raw_annotations = self.annotations[ix]
            aligned = label_set.get_aligned_label_ids_from_annotations(
                encoding, raw_annotations
            )
            aligned_labels.append(aligned)
        # END OF LABEL ALIGNMENT

        # MAKE A LIST OF TRAINING EXAMPLES. (This is where we add padding)
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
                        # The ids of the tokens
                        input_ids=encoding.ids[start:end]
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


class TraingingBatch:
    def __getitem__(self, item):
        return getattr(self, item)

    def __init__(self, examples: List[TrainingExample]):
        self.input_ids: torch.Tensor
        self.attention_masks: torch.Tensor
        self.labels: torch.Tensor
        input_ids: List[int] = []
        masks: List[int] = []
        labels: List[int] = []
        for ex in examples:
            input_ids.append(ex.input_ids)
            masks.append(ex.attention_masks)
            labels.append(ex.labels)
        self.input_ids = torch.LongTensor(input_ids)
        self.attention_masks = torch.LongTensor(masks)
        self.labels = torch.LongTensor(labels)



def prepare_batch_for_metrics(batch: TraingingBatch, predictions:torch.Tensor) -> Tuple[List[int], List[int]]:
    # get the sentence lengths
    s_lengths = batch.attention_masks.sum(dim=1)
    # iterate through the examples
    batch_true_values = []
    batch_pred_values = []
    for idx, length in enumerate(s_lengths):
        # get the true values
        true_values = batch.labels[idx][:length].tolist()
        batch_true_values.extend(true_values)
        # get the predicted values
        pred_values = torch.argmax(predictions, dim=2)[idx][:length].tolist()
        batch_pred_values.extend(pred_values)
    return batch_true_values, batch_pred_values

def bilu_to_non_bilu(iterat: Iterator) -> Dict[str, List[int]]:
    """Prepare non BILU labels mapping to ids"""
    tally = defaultdict(list)
    for i,item in enumerate(iterat):
        tally[item].append(i)
    return dict([(key,locs) for key,locs in tally.items()])

@dataclass
class BiluMappings:
    non_bilu_label_to_bilu_ids: Dict[str, Tuple[List[int], int]]
    non_bilu_label_to_id:  Dict[str, int]

# Tuple[Dict[int, Tuple[List[int], int]], Dict[str, int]]
def ids_to_non_bilu_label_mapping(labelset: LabelSet) -> BiluMappings:
    """Mapping from ids to BILU and non-BILU mapping. This is used to remove the BILU labels to regular labels"""
    target_names = list(labelset["ids_to_label"].values())
    wo_bilu = [bilu_label.split("-")[-1] for bilu_label in target_names]
    non_bilu_mapping = bilu_to_non_bilu(wo_bilu)

    BiluMappings.non_bilu_label_to_bilu_ids = {}
    BiluMappings.non_bilu_label_to_id = {}
    for target_name, labels_list in non_bilu_mapping.items():
        # 'upper_bound': ([1, 2, 3, 4], 1)
        BiluMappings.non_bilu_label_to_bilu_ids[target_name] = labels_list, labels_list[0]
        # 'upper_bound': 1
        BiluMappings.non_bilu_label_to_id[target_name] = labels_list[0]
    
    return BiluMappings

def remove_bilu_ids_from_true_and_pred_values(non_bilu_label_to_bilu_ids: Dict[int, Tuple[List[int], int]], 
    true_values: List[int], pred_values: List[int]) -> Tuple[List[int], List[int]]:
    """Reduce the BILI ids (true and predicted) to regular labels and ids"""

    for idx, label in enumerate(true_values):
        for _, (labels_list, non_bilu_label) in non_bilu_label_to_bilu_ids.items():
            if label in labels_list:
                true_values[idx] = int(non_bilu_label)
    
    for idx, label in enumerate(pred_values):
        for _, (labels_list, non_bilu_label) in non_bilu_label_to_bilu_ids.items():
            if label in labels_list:
                pred_values[idx] = int(non_bilu_label)

    return true_values, pred_values


def get_multilabel_metrics(true_values: List[int], pred_values: List[int],
    non_bilu_label_to_bilu_ids: Dict[str, Tuple[List[int], int]], non_bilu_label_to_id: Dict[str, int],
    labelset: LabelSet, remove_bilu: bool = True) -> Dict[str, Dict[str, Union[float, int]]]:
    """Create a classification report for all labels in the dataset"""
    
    if remove_bilu:
        labels = list(non_bilu_label_to_id.values())
        target_names = list(non_bilu_label_to_id.keys())
        true_values, pred_values = remove_bilu_ids_from_true_and_pred_values(non_bilu_label_to_bilu_ids, true_values, pred_values)
    else:
        labels = list(labelset["ids_to_label"].keys())
        target_names = list(labelset["ids_to_label"].values())

    metrics = classification_report(
        y_true=true_values, y_pred=pred_values, 
        labels=labels, target_names=target_names, output_dict=True, zero_division=0
    )
    return metrics
