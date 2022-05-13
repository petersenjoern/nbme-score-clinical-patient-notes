# %%
# Libraries
import pathlib
import yaml
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader
from torch.profiler import profile as tprofiler
from torch.profiler import schedule, tensorboard_trace_handler
from torch.utils.tensorboard import SummaryWriter
from utils.misc import seed_env, load_and_prepare_nbme_data, format_annotations_from_same_patient, LabelSet, CustomDataset, TraingingBatch
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
df_train, df_val = train_test_split(df_train, test_size=0.15)
# df_test = load_and_prepare_nbme_data(paths=cfg["datasets"], train=False)
data_train, unique_labels = format_annotations_from_same_patient(df_train)
data_val, unique_labels_val = format_annotations_from_same_patient(df_val)

label_set = LabelSet(labels=unique_labels)
num_labels = len(label_set.ids_to_label.values())
tokenizer = AutoTokenizer.from_pretrained(cfg.get("model").get("name"))
model_pretrained = BertForTokenClassification.from_pretrained(
    cfg.get("model").get("name"), num_labels=num_labels).to(DEVICE)
model_pretrained.config.id2label = label_set["ids_to_label"]
model_pretrained.config.label2id = label_set["labels_to_id"]
optimizer = AdamW(model_pretrained.parameters(), lr=0.0005)

# %%
trainset = CustomDataset(
    data=data_train, label_set=label_set, tokenizer=tokenizer)
trainloader = DataLoader(
    trainset,
    collate_fn=TraingingBatch,
    batch_size=cfg["hyper_params"].get("batch_size"),
    shuffle=cfg["hyper_params"].get("shuffle"),
)
# %%


def train(cfg: Dict[str, str], model: BertForTokenClassification, optimizer: optimization, 
    dataloader: DataLoader, labelset: LabelSet,
    save_directory: pathlib.Path = None) -> BertForTokenClassification:

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
            
            
            # pytorch profiler needs to be notified at each steps end
            prof.step() 

            # update the model one last time for this epoch
            optimizer.step()
            optimizer.zero_grad()

            # Log the metrics to every epoch
            tb.add_scalar("Loss", loss.item(), epoch)
    tb.add_hparams(
        {
            "lr": cfg["hyper_params"].get("learning_rate"), 
            "bsize": cfg["hyper_params"].get("batch_size"),
            "epochs":cfg["hyper_params"].get("epochs"), 
            "shuffle": cfg["hyper_params"].get("shuffle")},
        {
            #"precision": metrics['weighted avg']["precision"],
            #"recall": metrics['weighted avg']["recall"],
            #"f1-score": metrics['weighted avg']["f1-score"],
            "loss": loss.item(),
        },
    )
    if save_directory:
        model.save_pretrained(save_directory)
    return model


# %%
if __name__ == "__main__":
    model_finetuned = train(
        cfg=cfg, model=model_pretrained, 
        dataloader=trainloader, labelset=label_set, 
        optimizer=optimizer,
        save_directory=cfg["caching"]["finetuned_ner_model"]
    )
