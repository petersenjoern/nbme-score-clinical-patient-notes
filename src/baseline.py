# %%
# Libraries
import pathlib
import yaml
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

from utils.misc import seed_env, load_and_prepare_nbme_data, format_annotations_from_same_patient, LabelSet
from transformers import AutoTokenizer, AdamW, BertForTokenClassification, optimization
from typing import Dict, Iterator, List, Tuple, Union, Any
from dataclasses import dataclass

# Paths
PATH_BASE = pathlib.Path(__file__).absolute().parents[1]
PATH_YAML = PATH_BASE.joinpath("config.yaml")
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"


# Setup
with open(PATH_YAML, "r") as file:
    cfg = yaml.safe_load(file)

seed_env(424)


# %%
df_train = load_and_prepare_nbme_data(paths=cfg["datasets"], train=True)
data_train, unique_labels = format_annotations_from_same_patient(df_train)

label_set = LabelSet(labels=unique_labels)
num_labels = len(label_set.ids_to_label.values())
tokenizer = AutoTokenizer.from_pretrained(cfg.get("model").get("name"))
model_pretrained = BertForTokenClassification.from_pretrained(
    cfg.get("model").get("name"), num_labels=num_labels).to(DEVICE)
model_pretrained.config.id2label = label_set["ids_to_label"]
model_pretrained.config.label2id = label_set["labels_to_id"]
optimizer = AdamW(model_pretrained.parameters(), lr=0.0005)
# %%


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


# %%
trainset = CustomDataset(
    data=data_train, label_set=label_set, tokenizer=tokenizer)
trainloader = DataLoader(
    trainset,
    collate_fn=TraingingBatch,
    batch_size=cfg.get("hyperparams").get("batch_size"),
    shuffle=cfg.get("hyperparams").get("shuffle"),
)
# %%


def train(cfg: Dict[str, str], model: BertForTokenClassification, optimizer: optimization, dataloader: DataLoader, labelset: LabelSet,
          save_directory: pathlib.Path = None) -> BertForTokenClassification:

    for epoch in range(cfg["hyperparams"]["epochs"]):
        print("\nStart of epoch %d" % (epoch,))
        current_loss = 0
        epoch_true_sample_values = []
        epoch_pred_sample_values = []
        for step, batch in enumerate(dataloader):
            # move the batch tensors to the same device as the model
            batch.attention_masks = batch.attention_masks.to(DEVICE)
            batch.input_ids = batch.input_ids.to(DEVICE)
            batch.labels = batch.labels.to(DEVICE)
            # send 'input_ids', 'attention_mask' and 'labels' to the model
            outputs = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_masks,
                labels=batch.labels,
            )
            # the outputs are of shape (loss, logits)
            loss = outputs[0]
            # with the .backward method it calculates all
            # of  the gradients used for autograd
            loss.backward()
            # NOTE: if we append `loss` (a tensor) we will force the GPU to save
            # the loss into its memory, potentially filling it up. To avoid this
            # we rather store its float value, which can be accessed through the
            # `.item` method
            current_loss += loss.item()

            if step % 8 == 0 and step > 0:
                # update the model using the optimizer
                optimizer.step()
                # once we update the model we set the gradients to zero
                optimizer.zero_grad()
                # store the loss value for visualization
                print(current_loss)
                current_loss = 0

        # update the model one last time for this epoch
        optimizer.step()
        optimizer.zero_grad()

    if save_directory:
        model.save_pretrained(save_directory)
    return model


# %%
if __name__ == "__main__":
    model_finetuned = train(cfg=cfg, model=model_pretrained, dataloader=trainloader, labelset=label_set, optimizer=optimizer,
                            save_directory=cfg["caching"]["finetuned_ner_model"])
