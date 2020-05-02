import logging
import torch
from torch import nn
from nlplay.data.cache import DSManager, DS
from nlplay.features.text_cleaner import *
from nlplay.models.pytorch.classifiers.sru.sru_model import SRUClassifier
from nlplay.models.pytorch.dataset import DSGenerator
from nlplay.models.pytorch.trainer import PytorchModelTrainer

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG, datefmt="%Y-%m-%d %H:%M:%S")

# Input data files
ds = DSManager(DS.IMDB.value)
train_csv, test_csv, val_csv = ds.get_partition_paths()

# Inputs & Model Parameters
num_epochs = 20
batch_size = 32
ngram_range = (1, 1)
max_features = 20000
max_seq = 400
embedding_size = 100
hidden_size = 128
num_layers = 2
dropout = 0.2
lr = 0.001
num_workers = 1

# Data preparation
ds = DSGenerator()
train_ds, val_ds = ds.from_csv(train_file=train_csv, val_file=test_csv, ngram_range=ngram_range,
                               max_features=max_features, preprocess_func=base_cleaner, preprocess_ncore=3,
                               ds_max_seq=max_seq)

model = SRUClassifier(num_classes=ds.num_classes, vocabulary_size=ds.vocab_size,
                      embedding_size=embedding_size, dropout=dropout,
                      hidden_size=hidden_size, num_layers=num_layers)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = None

trainer = PytorchModelTrainer(model, criterion, optimizer, lr_scheduler=scheduler,
                              train_ds=train_ds, val_ds=val_ds,
                              batch_size=batch_size, n_workers=num_workers, epochs=num_epochs)
trainer.train_evaluate()

