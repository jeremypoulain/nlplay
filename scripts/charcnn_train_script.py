import numpy as np
import re
import csv
import logging
import torch
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from torch import nn
from torch.utils.data.dataset import TensorDataset
from nlplay.data.cache import DSManager, DS
from nlplay.models.pytorch.trainer import PytorchModelTrainer
from nlplay.models.pytorch.classifiers.charcnn import CharCNN_Zhang


logging.basicConfig(
    format="%(asctime)s %(message)s", level=logging.DEBUG, datefmt="%Y-%m-%d %H:%M:%S"
)

# Model parameters
num_epochs = 50
lr = 0.01
batch_size = 128
voc = """abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]\{\}"""
num_classes = 2
vocabulary_size = len(voc)
model_mode = "small"
conv_out_channels = 256
max_seq_len = 1014
dropout = 0.5
es_patience = 7
num_workers = 4

# Input data files
ds = DSManager(DS.IMDB.value)
train_csv, test_csv, val_csv = ds.get_partition_paths()

# Train/val Data setup
X_train = []
X_test = []
y_train = []
y_test = []
with open(train_csv, mode="r", encoding="utf-8") as f:
    reader = csv.DictReader(f, quotechar='"')
    for line in reader:
        X_train.append(line["sentence"])
        y_train.append(int(line["label"]))

with open(test_csv, mode="r", encoding="utf-8") as f:
    reader = csv.DictReader(f, quotechar='"')
    for line in reader:
        X_test.append(line["sentence"])
        y_test.append(int(line["label"]))

# Tokenisation
tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
tk.fit_on_texts(X_train)

# Vocabulary dictionary setup
char_dict = {}
for i, char in enumerate(voc):
    char_dict[char] = i + 1
tk.word_index = char_dict.copy()
tk.word_index[tk.oov_token] = max(char_dict.values()) + 1

# Convert words to vocab index
X_train = tk.texts_to_sequences(X_train)
X_test = tk.texts_to_sequences(X_test)

# Classes
y_train = np.asarray(y_train, int)
y_test = np.asarray(y_test, int)

# Padding
X_train = pad_sequences(X_train, maxlen=max_seq_len, padding='post')
X_train = np.array(X_train, dtype='float32')
X_test = pad_sequences(X_test, maxlen=max_seq_len, padding='post')
X_test = np.array(X_test, dtype='float32')

# Datasets creation
train_ds = TensorDataset(
    torch.from_numpy(X_train).long(), torch.from_numpy(y_train).long()
)
val_ds = TensorDataset(torch.from_numpy(X_test).long(), torch.from_numpy(y_test).long())

model = CharCNN_Zhang(
    num_classes=num_classes,
    vocabulary_size=vocabulary_size,
    model_mode=model_mode,
    max_seq_len=max_seq_len,
)

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = None
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer, mode="triangular2", base_lr=0.0001, max_lr=lr)

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
    early_stopping_patience=es_patience,
)
trainer.train_evaluate()

