import logging
import torch
from torch import nn
from nlplay.data.cache import DSManager, DS
from nlplay.features.text_cleaner import *
from nlplay.models.pytorch.classifiers.fasttext import PytorchFastText
from nlplay.models.pytorch.dataset import DSGenerator
from nlplay.models.pytorch.trainer import PytorchModelTrainer

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG, datefmt="%Y-%m-%d %H:%M:%S")

# Input data files
ds = DSManager(DS.IMDB.value)
train_csv, test_csv, val_csv = ds.get_partition_paths()

# Inputs & Model Parameters
num_epochs = 20
batch_size = 128
ngram_range = (1, 2)
max_features = 100000
max_seq = 1000
embedding_size = 50
dropout = 0.2
lr = 0.0025
num_workers = 1

# Data preparation
ds = DSGenerator()
train_ds, val_ds = ds.from_csv(train_file=train_csv, val_file=test_csv, ngram_range=ngram_range,
                               max_features=max_features, preprocess_func=base_cleaner, preprocess_ncore=3,
                               ds_max_seq=max_seq)

model = PytorchFastText(num_classes=ds.num_classes, vocabulary_size=ds.vocab_size,
                        embedding_size=embedding_size, drop_out=dropout)

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = None
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer, mode="triangular2", base_lr=0.0001, max_lr=lr)

trainer = PytorchModelTrainer(model, criterion, optimizer, lr_scheduler=scheduler,
                              train_ds=train_ds, val_ds=val_ds,
                              batch_size=batch_size, n_workers=num_workers, epochs=num_epochs)
trainer.train_evaluate()

# num_epochs = 10
# batch_size = 128
# ngram_range = (1, 2)
# max_features = 100000
# max_seq = 1000
# embedding_size = 50
# dropout = 0.20
# lr = 0.0025
# num_workers = 1
# 2020-04-27 23:21:35 ----------------------------------
# 2020-04-27 23:21:35 ---          SUMMARY           ---
# 2020-04-27 23:21:35 ----------------------------------
# 2020-04-27 23:21:35 Number of model parameters : 5000302
# 2020-04-27 23:21:35 Total Training Time: 0m 21s
# 2020-04-27 23:21:35 Total Time: 0m 21s
# 2020-04-27 23:21:35 Best Epoch: 9 - Accuracy Score: 0.884240
# 2020-04-27 23:21:35 ----------------------------------
