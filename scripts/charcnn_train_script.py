import logging
import string
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import TensorDataset
from nlplay.data.cache import DSManager, DS
from nlplay.features.text_cleaner import base_cleaner
from nlplay.models.pytorch.classifiers.charcnn import CharCNN_Zhang
from nlplay.models.pytorch.trainer import PytorchModelTrainer
from nlplay.models.pytorch.utils import char_vectorizer
from nlplay.utils.parlib import parallelApply

logging.basicConfig(
    format="%(asctime)s %(message)s", level=logging.DEBUG, datefmt="%Y-%m-%d %H:%M:%S"
)

# Input data files
ds = DSManager(DS.IMDB.value)
train_csv, test_csv, val_csv = ds.get_partition_paths()

# Vocabulary Setup
vocab = (
    list(string.ascii_lowercase)
    + list(string.digits)
    + list(string.punctuation)
    + ["\n"]
)
char2idx = {}
idx2char = {}
vocab = list(set(vocab))
for idx, t in enumerate(vocab):
    char2idx[t] = idx

# Experiment parameters
max_seq = 1014
vocabulary_size = len(vocab)
num_epochs = 100
lr = 0.0001
batch_size = 128
num_classes = 2
model_mode = "small"
dropout = 0.5
num_workers = 4


print("Importing data...")
df_train = pd.read_csv(train_csv, header=0)
df_test = pd.read_csv(test_csv, header=0)

print("Preprocessing...")
df_train["processed"] = parallelApply(df_train["sentence"], base_cleaner, 3)
df_test["processed"] = parallelApply(df_test["sentence"], base_cleaner, 3)
df_train = df_train.sample(frac=1, random_state=123).reset_index(drop=True)
x_train = list(df_train["processed"].values)
y_train = df_train["label"].values
x_test = list(df_test["processed"].values)
y_test = df_test["label"].values

print("Sentences vectorization...")
x_train = char_vectorizer(x_train, char2idx, max_seq)
x_test = char_vectorizer(x_test, char2idx, max_seq)
y_train = np.asarray(y_train, int)
y_test = np.asarray(y_test, int)

# Dataset creation
train_ds = TensorDataset(
    torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long()
)
val_ds = TensorDataset(
    torch.from_numpy(x_test).float(), torch.from_numpy(y_test).long()
)

model = CharCNN_Zhang(
    num_classes=num_classes,
    vocabulary_size=vocabulary_size,
    model_mode=model_mode,
    max_seq_len=max_seq,
    dropout=dropout
)

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = None

trainer = PytorchModelTrainer(
    model,
    criterion,
    optimizer,
    lr_scheduler=scheduler,
    train_ds=train_ds,
    val_ds=val_ds,
    batch_size=batch_size,
    n_workers=num_workers,
    epochs=num_epochs,
    early_stopping_patience=10,
)
trainer.train_evaluate()