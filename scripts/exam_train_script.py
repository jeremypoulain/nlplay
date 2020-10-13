import logging
import torch
from torch import nn
from nlplay.data.cache import DSManager, DS
from nlplay.features.text_cleaner import *
from nlplay.models.pytorch.classifiers.exam import EXAM
from nlplay.models.pytorch.dataset import DSGenerator
from nlplay.models.pytorch.trainer import PytorchModelTrainer

logging.basicConfig(
    format="%(asctime)s %(message)s", level=logging.DEBUG, datefmt="%Y-%m-%d %H:%M:%S"
)

# Input data files
ds = DSManager(DS.AG_NEWS.value)
train_csv, test_csv, val_csv = ds.get_partition_paths()

# Inputs & Model Parameters
num_epochs = 3
batch_size = 16
ngram_range = (1, 1)
region_size = 7
max_features = 100000
max_seq = 256
embedding_size = 128
dropout = 0.2
lr = 0.0001
num_workers = 1

# Data preparation
ds = DSGenerator()
train_ds, val_ds = ds.from_csv(
    text_col_idx=1,
    label_col_idx=0,
    train_file=train_csv,
    val_file=test_csv,
    ngram_range=ngram_range,
    max_features=max_features,
    preprocess_func=base_cleaner,
    preprocess_ncore=3,
    ds_max_seq=max_seq,
)

model = EXAM(
    num_classes=ds.num_classes,
    vocabulary_size=ds.vocab_size,
    embedding_size=embedding_size,
    region_size=region_size,
    max_sent_len=max_seq,
    activation_function="relu",
    drop_out=dropout,
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
)
trainer.train_evaluate()
