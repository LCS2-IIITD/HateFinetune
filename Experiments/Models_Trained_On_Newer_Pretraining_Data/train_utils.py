import os
import evaluate
from datasets import load_dataset
import pandas as pd
from ds import *
import torch
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding,AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer, AutoModel, AutoConfig
from custom_dataset import CustomDataset
from model import ClassifierOnMiddleLayer
from transformers import AdamW, get_scheduler
from torch.utils.data import DataLoader
from datasets import load_metric
import json
from tqdm.auto import tqdm
import pickle
from sklearn.metrics import classification_report
import random
import numpy as np
from sklearn.metrics import classification_report

def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999
    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def prepare_data_loaders(tokenizer, dataset_hf_tokenized):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataset = dataset_hf_tokenized['train']
    val_dataset = dataset_hf_tokenized['validation']
    test_dataset = dataset_hf_tokenized['test']

    train_dataset = CustomDataset(train_dataset)
    val_dataset = CustomDataset(val_dataset)
    test_dataset = CustomDataset(test_dataset)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=16)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=16)
    
    return train_dataloader, val_dataloader, test_dataloader

def perform_eval_on_validation(val_dataloader, device, model, epoch, progress_bar_eval):
    all_preds = []
    true_labels = []
    for step, batch in enumerate(val_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        predictions = outputs.logits.argmax(-1)
        progress_bar_eval.update(1)
        all_preds.append(predictions)
        true_labels.append(batch['label'])

    # Convert the predictions to a numpy array
    ls_preds = []
    for i in all_preds:
        for j in i:
            ls_preds.append(j.item())
            
    ls_true_labels = []
    for i in true_labels:
        for j in i:
            ls_true_labels.append(j.item())
    
    cr = classification_report(ls_true_labels, ls_preds)
    
    # Log the metrics to metric_values
    return_val = {
        'epoch': epoch,
        'true_labels': ls_true_labels,
        'all_preds': ls_preds,
        'classification_report': cr
    }

    return return_val

def perform_eval_on_test(test_dataloader, device, model, epoch, progress_bar_eval):
    all_preds = []
    true_labels = []
    for step, batch in enumerate(test_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        predictions = outputs.logits.argmax(-1)
        progress_bar_eval.update(1)
        all_preds.append(predictions)
        true_labels.append(batch['label'])

    # Convert the predictions to a numpy array
    ls_preds = []
    for i in all_preds:
        for j in i:
            ls_preds.append(j.item())
            
    ls_true_labels = []
    for i in true_labels:
        for j in i:
            ls_true_labels.append(j.item())
    
    cr = classification_report(ls_true_labels, ls_preds)
    
    # Log the metrics to metric_values
    return_val = {
        'epoch': epoch,
        'true_labels': ls_true_labels,
        'all_preds': ls_preds,
        'classification_report': cr
    }

    return return_val