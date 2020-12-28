import logging
import torch
from torch import nn
from nlplay.data.cache import DSManager, DS
from nlplay.features.text_cleaner import *
from nlplay.models.pytorch.classifiers.leam import LEAM
from nlplay.models.pytorch.dataset import DSGenerator
from nlplay.models.pytorch.trainer import PytorchModelTrainer

logging.basicConfig(
    format="%(asctime)s %(message)s", level=logging.DEBUG, datefmt="%Y-%m-%d %H:%M:%S"
)

# Input data files
ds = DSManager(DS.IMDB.value)
train_csv, test_csv, val_csv = ds.get_partition_paths()

# Inputs & Model Parameters
num_epochs = 20
batch_size = 64
ngram_range = (1, 1)
max_seq = 150
max_features = 25000
embedding_size = 100
fc_hidden_sizes = [50]
fc_activation_functions = ["relu"]
fc_dropouts = [0.5]
ngram = 55
lr = 0.0025
num_workers = 1

# Data preparation
ds = DSGenerator()
train_ds, val_ds = ds.from_csv(
    train_file=train_csv,
    val_file=test_csv,
    ngram_range=ngram_range,
    max_features=max_features,
    preprocess_func=base_cleaner,
    preprocess_ncore=3,
    ds_max_seq=max_seq,
)

model = LEAM(
    num_classes=ds.num_classes,
    vocabulary_size=ds.vocab_size,
    embedding_size=embedding_size,
    fc_hidden_sizes=fc_hidden_sizes,
    fc_activation_functions=fc_activation_functions,
    fc_dropouts=fc_dropouts,
    ngram=ngram,
    apply_sm=False,
)

criterion = nn.CrossEntropyLoss()
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

# num_epochs = 20
# batch_size = 64
# ngram_range = (1, 1)
# embedding_size = 100
# max_seq = 150
# max_features = 25000
# dropout = 0.2
# lr = 0.0025
# num_workers = 1
# 2020-12-28 15:39:47 ------------------------------------------
# 2020-12-28 15:39:47 ---              SUMMARY               ---
# 2020-12-28 15:39:47 ------------------------------------------
# 2020-12-28 15:39:47 Number of model parameters : 2501248
# 2020-12-28 15:39:47 Total Training Time: 0m 13s
# 2020-12-28 15:39:47 Total Time: 0m 13s
# 2020-12-28 15:39:47 Best Epoch: 1 - Accuracy Score: 0.877600
# 2020-12-28 15:39:47 ------------------------------------------
