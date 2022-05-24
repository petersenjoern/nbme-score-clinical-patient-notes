#%%
import argparse
import os
import sys
import gc
import copy
import time
import random
import string
import pathlib
import yaml

# For data manipulation
import numpy as np
import pandas as pd

# Pytorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# Utils
from tqdm import tqdm
from collections import defaultdict
import itertools

# Sklearn Imports
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold

# For Transformer Models
from transformers import AutoTokenizer, AutoModel, AutoConfig, AdamW, AutoModelForQuestionAnswering
from transformers import RobertaTokenizerFast, DebertaV2TokenizerFast, DebertaV2Tokenizer
from tokenizers import AddedToken

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

os.environ['TOKENIZERS_PARALLELISM'] = "false"

device = torch.device('cuda')
#%%

# Constants
DEBUG = False

PATH_BASE = pathlib.Path(__file__).absolute().parents[1]
PATH_YAML = PATH_BASE.joinpath("config.yaml")



model_coeff = {
    'nbme_182': 0.40 * 0.55 * 0.5,
    'nbme_182_full':  0.60 * 0.55 * 0.5,
    'nbme_254': 0.40 * 0.55 * 0.5,
    'nbme_254_full':  0.60 * 0.55 * 0.5,
    'nbme_187': 0.40 * 0.45 * 0.60,
    'nbme_187_full':  0.60 * 0.45 * 0.60,
    'nbme_256': 0.40 * 0.45 * 0.40,
    'nbme_256_full':  0.60 * 0.45 * 0.40,
}

pred_thr = -0.20


#%%

# Load Config
with open(PATH_YAML, "r") as file:
    cfg = yaml.safe_load(file)

cfg["datasets"]

#%%

if DEBUG:
    test = pd.read_csv(cfg["datasets"]["test"])[:2000]
else:
    test = pd.read_csv(cfg["datasets"]["test"])
# %%
features = pd.read_csv(cfg["datasets"]["features"])
feature_year = features.loc[[ft.endswith('year') for ft in features.feature_text]]
feature_year = feature_year.feature_num.values
feature_female = features.loc[[ft == 'Female' for ft in features.feature_text]]
feature_female = feature_female.feature_num.values
feature_male = features.loc[[ft == 'Male' for ft in features.feature_text]]
feature_male = feature_male.feature_num.values

# %%
notes = pd.read_csv(cfg["datasets"]["patient_notes"])
test_notes = notes[notes.pn_num.isin(set(test.pn_num.unique()))].reset_index(drop=True)
test_notes['text_length'] = [len(pn_history) for pn_history in test_notes.pn_history]
test = test.merge(test_notes[['pn_num', 'text_length']], how='left', on='pn_num')

# %%
class NBMEDataset(Dataset):
    def __init__(self, annotations, CONFIG, features, notes,):
        super(NBMEDataset, self).__init__()
        self.id = annotations.id.values
        self.pn_num = annotations.pn_num.values
        self.feature_num = annotations.feature_num.values
        try:
            self.location = annotations.location.values
        except:
            self.location = None
        self.feature_token = self.tokenize(features, 'feature_num', 'feature_text', CONFIG)
        
        self.pn_history_token = self.tokenize(notes[notes.pn_num.isin(set(annotations.pn_num.unique()))], 
                                              'pn_num', 'pn_history', CONFIG)
        self.max_length = CONFIG['max_length']
        tokenizer = CONFIG['tokenizer']
        self.special_tokens = {
            "sep": tokenizer.sep_token_id,
            "cls": tokenizer.cls_token_id,
            "pad": tokenizer.pad_token_id,            
        }
        self.config = CONFIG
        
    def __len__(self):
        return len(self.pn_num)
    
    def __getitem__(self, idx):        
        pn_num = self.pn_num[idx]
        pn_history_token = self.pn_history_token[pn_num]
        feature_num = self.feature_num[idx]
        feature_token = self.feature_token[feature_num]
        location = None
        if self.location is not None:
            location = self.location[idx]
        data = self.get_data(pn_history_token, feature_token, location, feature_num)
        data.update({
            'id':self.id[idx],
        })
        return data
    
    def get_data(self, pn_history_token, feature_token, location, feature_num, ):
        max_length = self.max_length
        text = pn_history_token['text']
        pn_history_token = pn_history_token['tokens']
        feature_token = feature_token['tokens']
        
        sep = self.special_tokens["sep"]
        cls = self.special_tokens["cls"]
        pad = self.special_tokens["pad"]
        q_input_ids = [cls] + feature_token['input_ids'] + [sep]
        if "roberta" in self.config['model_name']:
            q_input_ids = q_input_ids + [sep]       
        input_ids = q_input_ids + pn_history_token['input_ids']
        input_ids = input_ids[: max_length - 1] + [self.special_tokens["sep"]]
        len_token = len(input_ids)
        
        offset_mapping = [(0,0)] * len(q_input_ids) + pn_history_token['offset_mapping']
        offset_mapping = offset_mapping[: max_length - 1] + [(0,0)]
        max_token = len(text)
        assert(len_token == len(offset_mapping))
        
        len_padding = max_length - len_token
        if len_padding > 0:
            input_ids = input_ids + [self.special_tokens["pad"]] * len_padding
            
        attention_mask = np.zeros(max_length, dtype='int')
        attention_mask[:len_token] = 1
        
        if "roberta" in self.config['model_name']:
            token_type_ids = [0]
        else:
            token_type_ids = np.ones(max_length)
            token_type_ids[:len(q_input_ids)] = 0
            
        out_dict = {
            'input_ids' : torch.tensor(input_ids, dtype=torch.long),
            'token_type_ids' :  torch.tensor(token_type_ids, dtype=torch.long),
            'offset_mapping' : offset_mapping,
            'attention_mask' : torch.tensor(attention_mask, dtype=torch.long),
            'len_token' : len_token,
            'max_token' : max_token,
            'text' : text,
            'feature_num' : feature_num,
        }  
        
        if self.location is not None:
            len_text = len(text)
            char_type = np.zeros((len_text, ))
            char_start = np.zeros((len_text, ))
            char_end = np.zeros((len_text, ))
            annots = eval(location)
            annots = [a for ans in annots for a in ans.split(';')]
            for annot in annots:
                annot = annot.split()            
                start = int(annot[0])
                end = int(annot[1])
                #print(feature, row['feature_num'], start, end)
                char_type[start:end] = 1
                char_start[start] = 1
                char_end[end - 1] = 1
            token_type = np.zeros((max_length, ))
            token_start = np.zeros((max_length, ))
            token_end = np.zeros((max_length, ))
            for i, (start, end) in enumerate(offset_mapping):
                if start == end:
                    continue
                token_type[i] = char_type[start:end].max(0)
                token_start[i] = char_start[start:end].max(0)
                token_end[i] = char_end[start:end].max(0)
            out_dict.update({
                'token_type':torch.tensor(token_type, dtype=torch.float32).unsqueeze(-1),
                'token_start':torch.tensor(token_start, dtype=torch.float32).unsqueeze(-1),
                'token_end':torch.tensor(token_end, dtype=torch.float32).unsqueeze(-1),
            })
        
        return out_dict
    
    def tokenize(self, data, key, text, CONFIG):
        res = {k:({'tokens':CONFIG['tokenizer'](t, 
                                                return_offsets_mapping=True, 
                                                return_attention_mask=False,
                                                add_special_tokens=False, 
                                                max_length=CONFIG['max_length'], 
                                                truncation=True,
                                                padding=False,
                                               ), 
                   'text':t,
                  }) for k,t in zip(data[key], data[text])}
        return res

# %%
not_collate_keys = ['id', 'offset_mapping', 'len_token', 'max_token', 'text', 'feature_num']
collate_keys = ['input_ids', 'token_type_ids', 'attention_mask', 
                #'token_type', 'token_start', 'token_end',
               ]

def feedback_collate(batch):
    batch_dict = {}
    len_token_max = np.max([sample['len_token'] for sample in batch])
    for key in collate_keys:
        try:
            batch_dict[key] = torch.stack([b[key][:len_token_max] for b in batch])
        except:
            print('key not found:', key)
    for key in not_collate_keys:
        if key == 'offset_mapping':
            batch_dict[key] = [b[key][:len_token_max] for b in batch]
        else:
            batch_dict[key] = [b[key] for b in batch]
    return batch_dict


#%%

def get_data_loader(data, shuffle, CONFIG, features=features, notes=notes,):
    if shuffle:
        batch_size = CONFIG['train_batch_size']
    else:
        batch_size = CONFIG['valid_batch_size']
    dataset = NBMEDataset(data, CONFIG, features, notes,)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=CONFIG['workers'],
        shuffle=shuffle,
        pin_memory=True,
        #worker_init_fn=worker_init_fn,
        collate_fn=feedback_collate,
    )
    return data_loader


# %%

def criterion(pred, target):
    return nn.BCEWithLogitsLoss(reduction='none')(pred, target)

#%%
def glorot_uniform(parameter):
    nn.init.xavier_uniform_(parameter.data, gain=1.0)
    
class NBMEHead(nn.Module):
    def __init__(self, input_dim, output_dim, loss, criterion):
        super(NBMEHead, self).__init__()
        self.loss = loss
        self.criterion = criterion
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.classifier = nn.Linear(input_dim, output_dim)
        glorot_uniform(self.classifier.weight)
        
    def forward(self, x, attention_mask, target=None):
        # x is B x S x C
        logits1 = self.classifier(self.dropout1(x))
        logits2 = self.classifier(self.dropout2(x))
        logits3 = self.classifier(self.dropout3(x))
        logits4 = self.classifier(self.dropout4(x))
        logits5 = self.classifier(self.dropout5(x))
                              
        logits = ((logits1 + logits2 + logits3 + logits4 + logits5) / 5)
        logits = logits * attention_mask
        
        if self.loss:
            loss1 = self.criterion(logits1, target)
            loss2 = self.criterion(logits2, target)
            loss3 = self.criterion(logits3, target)
            loss4 = self.criterion(logits4, target)
            loss5 = self.criterion(logits5, target)
            loss = (loss1 + loss2 + loss3  + loss4 + loss5) / 5
            
            #print(loss.shape, attention_mask.shape)
            loss = loss * attention_mask
            loss = loss.sum(1) / (1e-6 + attention_mask.sum(1))
            loss = loss.mean()
        else:
            loss = 0
        return logits, loss

# %%
prediction_keys = ['token_type_logits',]
loss_keys = ['loss', 'tp_count', 'all_count', ]
not_collate_keys, collate_keys


# %%

keep_collate_keys = []
test_collate_keys =  ['input_ids', 'attention_mask', 'token_type_ids']


# %%
spaces = ' \n\r'

def post_process_spaces(pred, text):
    text = text[:len(pred)]
    pred = pred[:len(text)]
    if text[0] in spaces:
        pred[0] = 0
    if text[-1] in spaces:
        pred[-1] = 0

    for i in range(1, len(text) - 1):
        if text[i] in spaces:
            if pred[i] and not pred[i - 1]:  # space before
                pred[i] = 0

            if pred[i] and not pred[i + 1]:  # space after
                pred[i] = 0

            if pred[i - 1] and pred[i + 1]:
                pred[i] = 1
 
    return pred
# %%


def pred_to_chars(token_type_logits, len_token, max_token, offset_mapping, text, feature_num):
    token_type_logits = token_type_logits[:len_token]
    offset_mapping = offset_mapping[:len_token]
    char_preds = np.ones(len(text)) * -1e10
    for i, (start,end) in enumerate(offset_mapping):
        char_preds[start:end] = token_type_logits[i]
    return (char_preds, text)


# %%
def pred_to_chars(token_type_logits, len_token, max_token, offset_mapping, text, feature_num):
    token_type_logits = token_type_logits[:len_token]
    offset_mapping = offset_mapping[:len_token]
    char_preds = np.ones(len(text)) * -1e10
    for i, (start,end) in enumerate(offset_mapping):
        if text[start:end] == 'of' and start > 0 and text[start-1:end] == 'yof':
            if feature_num in feature_female:
                char_preds[end-1:end] = 1
            elif feature_num in feature_year:
                char_preds[start:start+1] = token_type_logits[i-1]
            else:
                char_preds[start:end] = token_type_logits[i]
        elif text[start:end] == 'om' and start > 0 and text[start-1:end] == 'yom':
            if feature_num in feature_male:
                char_preds[end-1:end] = 1
            elif feature_num in feature_year:
                char_preds[start:start+1] = token_type_logits[i-1]
            else:
                char_preds[start:end] = token_type_logits[i]
        else:
            char_preds[start:end] = token_type_logits[i]
    return (char_preds, text)
# %%


def char_preds_to_string(char_preds, text, pred_thr):
    char_preds = (char_preds > pred_thr) * 1
    post_process_spaces(char_preds, text)
    indices = np.where(char_preds == 1)[0]
    indices_grouped = [
        list(g) for _, g in itertools.groupby(
            indices, key=lambda n, c=itertools.count(): n - next(c)
        )
    ]
    spans = [f"{min(r)} {max(r) + 1}" for r in indices_grouped]
    spans = ';'.join(spans)
    return spans


# %%


def test_epoch(loader, models, device):

    for model in models:
        model.eval()
    char_preds = []
    with torch.no_grad():
        if CONFIG['verbose']:
            bar = tqdm(range(len(loader)))
        else:
            bar = range(len(loader))
        load_iter = iter(loader)

        for i in bar:
            batch = load_iter.next()
            input_dict = {k:batch[k].to(device, non_blocking=True) for k in collate_keys}

            batch_out_dict = {}
            for key in prediction_keys :
                batch_out_dict[key] = 0             
            for model in models:
                out_dict = model(input_dict)
                for key in prediction_keys :
                    batch_out_dict[key] = batch_out_dict[key] + out_dict[key].detach() / len(models)             
            token_type_logits = (batch_out_dict['token_type_logits']).detach()
            token_type_logits = token_type_logits.cpu().numpy()

            char_preds.extend([
                pred_to_chars(*p) for p in zip(token_type_logits, 
                                                batch['len_token'], 
                                                batch['max_token'], 
                                                batch['offset_mapping'],
                                                batch['text'],
                                                batch['feature_num'],
                                               )
            ])

    return char_preds


# %%


def load_model_checkpoint(dirname, fname, fold):
    model = NBMEModel(CONFIG['model_name'], loss=False, pretrained=False).to(device)
    checkpoint = torch.load('../input/%s/%s_%d.pt' % (dirname, fname, fold))
    print(dirname, fname, fold, checkpoint['epoch'])
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

def get_char_preds(test, CONFIG, notes):
    device = torch.device('cuda')
    
    # models = [load_model_checkpoint(CONFIG['dirname'],CONFIG['fname'], fold) for fold in CONFIG["folds"]]
    model_1=AutoModel.from_pretrained(CONFIG['model_name'])
    models = [model_1]

    test_data_loader = get_data_loader(test, shuffle=False, CONFIG=CONFIG, notes=notes)
    char_preds = test_epoch(test_data_loader, models, device)   
    del test_data_loader, models
    gc.collect()
    return char_preds

def get_preds(char_preds, texts, pred_thr):
    preds = [char_preds_to_string(p, text, pred_thr) for p,text in zip(char_preds, texts)]
    df = pd.DataFrame({'id':test['id'], 'location':preds,})
    return df


# %%
char_preds_all = []
model_coeff_all = []

# %%
class NBMEModel(nn.Module):
    def __init__(self, model_name, loss=False, pretrained=True):
        super(NBMEModel, self).__init__()
        config = CONFIG['config']
        self.config = config
        if pretrained:
            self.backbone = AutoModel.from_pretrained(model_name)
        else:
            self.backbone = AutoModel.from_config(config)
        self.backbone.resize_token_embeddings(len(CONFIG['tokenizer']))
        self.loss = loss
        #self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout0 = nn.Dropout(p=CONFIG['dropout'])
        self.rnn0 = nn.LSTM(CONFIG['config'].hidden_size,
                           CONFIG['config'].hidden_size//2,
                           num_layers=1,
                           batch_first=True,
                           bidirectional=True,
                          )
        self.token_type_head = NBMEHead(config.hidden_size, 1,
                                           loss, criterion)
        self.token_start_head = NBMEHead(config.hidden_size, 1,
                                           loss, criterion)
        self.token_end_head = NBMEHead(config.hidden_size, 1,
                                           loss, criterion)
        self.model_name = model_name
        weight_data = torch.linspace(-5, 5, 1+self.config.num_hidden_layers)
        weight_data = weight_data.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.weights = nn.Parameter(weight_data, requires_grad=True)
       
    def forward(self, input_dict):     
        input_ids = input_dict['input_ids']
        attention_mask = input_dict['attention_mask']
        if 'roberta' in self.model_name:
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                                output_hidden_states=True)
        else:
            token_type_ids = input_dict['token_type_ids']
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids, output_hidden_states=True)
        x = torch.stack(out.hidden_states)
        w = F.softmax(self.weights, 0)
        w = self.dropout0(w)
        x = (w * x).sum(0)
        x = self.rnn0(x)[0]
        #x = out.hidden_states[-1]
        if self.loss:
            token_type = input_dict['token_type']
            token_start = input_dict['token_start']
            token_end = input_dict['token_end']
            target_mask = (attention_mask * token_type_ids).unsqueeze(-1)
        else:
            token_type = None
            token_start = None
            token_end = None
            target_mask = 1
        token_type_logits, loss_type = self.token_type_head(x, target_mask, token_type) 
        token_start_logits, loss_start = self.token_start_head(x, target_mask, token_start) 
        token_end_logits, loss_end = self.token_end_head(x, target_mask, token_end)
        out_dict = {
            'token_type_logits' : token_type_logits,
            'token_start_logits' : token_start_logits,
            'token_end_logits' : token_end_logits,
        }
        
        if self.loss:
            loss = loss_type + loss_start + loss_end
            token_type_pred = ((token_type_logits >= 0) * target_mask).detach()
            tp_count = (token_type_pred * token_type).sum().detach().item()
            all_count = (token_type_pred.sum() + token_type.sum()).detach().item()
            #print(tp_count, token_type_pred.sum().item(), token_type.sum().item())
            out_dict.update({
                'loss' : loss,
                'tp_count' : tp_count,
                'all_count' : all_count,
            })
            
        return out_dict
# %%
def clean_abbrev(text):
    text = text.replace('FHx', 'FH ')
    text = text.replace('FHX', 'FH ')
    text = text.replace('PMHx', 'PMH ')
    text = text.replace('PMHX', 'PMH ')
    text = text.replace('SHx', 'SH ')
    text = text.replace('SHX', 'SH ')
    text = text.lower()
    return text


# %%



CONFIG = {"fname" : 'nbme_254',
          'dirname' : 'nbme-254',
          "seed": 2021,
          "epochs": 10,
          "model_name": "microsoft/deberta-v2-xlarge",
          "train_batch_size": 8,
          "valid_batch_size": 32,
          "max_length": 512,
          "learning_rate": 1e-4,
          "scheduler": 'CosineAnnealingLR',
          "min_lr": 1e-8,
          "folds": [0, 1, 2, 3],
          "n_accumulate": 4,
          "num_classes": 2,
          "margin": 0.5,
          "device": torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
          'opt_wd_non_norm_bias' : 0.01,
          'opt_wd_norm_bias' : 0, # same as Adam in Fastai
          'opt_beta1' : 0.9,
          'opt_beta2' : 0.99,
          'opt_eps' : 1e-5, # same as Adam in Fastai
          'verbose': DEBUG,
          'valid_check' : 8,
          'workers':8,
          'dropout' : 0.2,
          'checkpointing':False,
          }

if CONFIG['fname'] in model_coeff:
    model_coeff_all.append(model_coeff[CONFIG['fname'] ])
    CONFIG["config"] = AutoConfig.from_pretrained(CONFIG['model_name'])
    tokenizer = DebertaV2TokenizerFast.from_pretrained(CONFIG['model_name'])
    CONFIG["tokenizer"] = tokenizer

    lf_token = AddedToken("\r\n", lstrip=True, rstrip=True)
    fh_token = AddedToken("fh", single_word=True, lstrip=False, rstrip=True)
    pmh_token = AddedToken("pmh", single_word=True, lstrip=False, rstrip=True)
    sh_token = AddedToken("sh", single_word=True, lstrip=False, rstrip=True)
    tokenizer.add_tokens([lf_token])
    
    test_clean_notes = test_notes.copy()
    test_clean_notes['pn_history'] = [clean_abbrev(text) for text in test_clean_notes['pn_history'] ]
    char_preds = get_char_preds(test, CONFIG, test_clean_notes)
    char_preds_all.append(char_preds)


# %%
