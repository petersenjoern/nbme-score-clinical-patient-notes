# %%
# Libraries
import pathlib
import yaml
import torch
import json
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader
from torch.profiler import profile as tprofiler
from torch.profiler import schedule, tensorboard_trace_handler
from torch.utils.tensorboard import SummaryWriter
from utils.misc import (seed_env, load_and_prepare_nbme_data, 
    format_annotations_from_same_patient, LabelSet, CustomDataset, 
    TraingingBatch, prepare_batch_for_metrics, 
    ids_to_non_bilu_label_mapping, get_multilabel_metrics, BiluMappings
)
from transformers import AutoTokenizer, AdamW, BertForTokenClassification, optimization
from typing import Dict

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
df_train, df_val = train_test_split(df_train, test_size=0.1)
# df_test = load_and_prepare_nbme_data(paths=cfg["datasets"], train=False)
data_train, unique_labels = format_annotations_from_same_patient(df_train)
data_val, unique_labels_val = format_annotations_from_same_patient(df_val)
[print(label) for label in unique_labels_val if not label in unique_labels]

# prepare labelset
label_set = LabelSet(labels=unique_labels)
num_labels = len(label_set.ids_to_label.values())

# prepare tokenizer, model, optimizer
tokenizer = AutoTokenizer.from_pretrained(cfg.get("model").get("name"))
model_pretrained = BertForTokenClassification.from_pretrained(
    cfg.get("model").get("name"), num_labels=num_labels).to(DEVICE)
model_pretrained.config.id2label = label_set["ids_to_label"]
model_pretrained.config.label2id = label_set["labels_to_id"]
optimizer = AdamW(model_pretrained.parameters(), lr=0.0005)

# prepare bilu mappings
bilu_mappings = ids_to_non_bilu_label_mapping(label_set)

# %%
trainset = CustomDataset(
    data=data_train, label_set=label_set, tokenizer=tokenizer)
trainloader = DataLoader(
    trainset,
    collate_fn=TraingingBatch,
    batch_size=cfg["hyper_params"].get("batch_size"),
    shuffle=cfg["hyper_params"].get("shuffle"),
)

valset = CustomDataset(
    data=data_val, label_set=label_set, tokenizer=tokenizer)
valloader = DataLoader(
    valset,
    collate_fn=TraingingBatch,
    batch_size=cfg["hyper_params"].get("batch_size"),
    shuffle=False
)
# %%


def train(cfg: Dict[str, str], model: BertForTokenClassification, optimizer: optimization, 
    dataloader: DataLoader, labelset: LabelSet, bilu_mappings: BiluMappings,
    save_directory: pathlib.Path = None) -> BertForTokenClassification:
    """Train NLP Classification model"""

    model_comment = (f'bsize: {cfg["hyper_params"].get("batch_size")}, lr: {cfg["hyper_params"].get("learning_rate")}, '
        f'epochs: {cfg["hyper_params"].get("epochs")}, shuffle: {cfg["hyper_params"].get("shuffle")} device: {DEVICE}')
    tb = SummaryWriter(
        log_dir=cfg.get("caching").get("tensorboard_metrics"), 
        filename_suffix="-summary", 
        comment=model_comment
        )
    with tprofiler(
            schedule=schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=tensorboard_trace_handler(cfg.get("caching").get("tensorboard_profiler")),
            record_shapes=True,
            with_stack=True,
    ) as prof:
        for epoch in range(cfg["hyper_params"]["epochs"]):
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
            
                # Log every 250 batches.
                if step % 250 == 0:
                    batch_true_values, batch_pred_values = prepare_batch_for_metrics(batch=batch, predictions=outputs[1])
                    epoch_true_sample_values.extend(batch_true_values)
                    epoch_pred_sample_values.extend(batch_pred_values)

            # pytorch profiler needs to be notified at each steps end
            prof.step() 

            # update the model one last time for this epoch
            optimizer.step()
            optimizer.zero_grad()

            # visually inspect if metrics are improving over time
            metrics = get_multilabel_metrics(
                epoch_true_sample_values,
                epoch_pred_sample_values,
                bilu_mappings.non_bilu_label_to_bilu_ids,
                bilu_mappings.non_bilu_label_to_id,
                labelset,
                cfg["model"]["evaluation"]["remove_bilu"]
            )
            # Log the metrics to every epoch
            tb.add_scalar("Loss", loss.item(), epoch)
            tb.add_scalar("Precision", metrics['weighted avg']["precision"], epoch)
            tb.add_scalar("Recall", metrics['weighted avg']["recall"], epoch)
            tb.add_scalar("F1-Score", metrics['weighted avg']["f1-score"], epoch)
    
    # record the final results with hyperparams used
    tb.add_hparams(
        {
            "lr": cfg["hyper_params"]["learning_rate"], "bsize": cfg["hyper_params"]["batch_size"],
            "epochs":cfg["hyper_params"]["epochs"], "shuffle": cfg["hyper_params"]["shuffle"]},
        {
            "precision": metrics['weighted avg']["precision"],
            "recall": metrics['weighted avg']["recall"],
            "f1-score": metrics['weighted avg']["f1-score"],
            "loss": loss.item(),
        },
    )
    if save_directory:
        print("saving model ..")
        torch.save(model, pathlib.Path(save_directory).joinpath("bert_finetuned.pt"))
        print("finished saving model")
    return model


def evaluate(cfg: Dict[str, str], model: BertForTokenClassification,
    dataloader: DataLoader, labelset:LabelSet, bilu_mappings: BiluMappings,
    load_directory:pathlib.Path=None, save_directory:pathlib.Path=None) -> None:
    """ Evaluate the Model"""
    if not model:
        if load_directory:
            try:
                model = torch.load(load_directory)
            except:
                raise Exception("Model couldnt be loaded under: %s", load_directory)
        
    # Evaluate on test dataset
    model = model.eval() #equivalent to model.train(False)
    epoch_true_sample_values = []
    epoch_pred_sample_values = []
    for step, batch in enumerate(dataloader):
        # do not calculate the gradients
        with torch.no_grad():
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
            batch_true_values, batch_pred_values = prepare_batch_for_metrics(batch=batch, predictions=outputs[1])
            epoch_true_sample_values.extend(batch_true_values)
            epoch_pred_sample_values.extend(batch_pred_values)
    metrics = get_multilabel_metrics(
        epoch_true_sample_values,
        epoch_pred_sample_values,
        bilu_mappings.non_bilu_label_to_bilu_ids,
        bilu_mappings.non_bilu_label_to_id,
        labelset,
        cfg["model"]["evaluation"]["remove_bilu"]
    )
    if save_directory:
        with open(save_directory, 'w') as outfile:
            json.dump(metrics, outfile)
    print(metrics)

# %%
if __name__ == "__main__":
    model_finetuned = train(
        cfg=cfg,
        model=model_pretrained, 
        dataloader=trainloader, 
        labelset=label_set, 
        optimizer=optimizer,
        bilu_mappings=bilu_mappings,
        save_directory=cfg["caching"]["finetuned_ner_model"]
    )
    evaluate(
        cfg=cfg,
        model=model_finetuned,
        dataloader=valloader,
        labelset=label_set,
        bilu_mappings=bilu_mappings,
        save_directory=cfg["caching"]["finetuned_ner_metrics"],
        load_directory=pathlib.Path(cfg["caching"]["finetuned_ner_model"]))
