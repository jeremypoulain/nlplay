import logging
import torch
from torch import nn
from nlplay.features.text_cleaner import *
from nlplay.models.pytorch.classifiers.fasttext import PytorchFastText
from nlplay.models.pytorch.dataset import DSGenerator
from nlplay.models.pytorch.trainer import PytorchModelTrainer

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG, datefmt="%Y-%m-%d %H:%M:%S")

# Inputs & Model Parameters
train_csv = "../nlplay/data_cache/IMDB/IMDB_train.csv"
test_csv = "../nlplay/data_cache//IMDB/IMDB_test.csv"

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
trainer = PytorchModelTrainer(model, criterion, optimizer,
                              train_ds=train_ds, test_ds=val_ds,
                              batch_size=batch_size, n_workers=num_workers, epochs=num_epochs)
trainer.train()

# num_epochs = 5
# batch_size = 128
# ngram_range = (1, 2)
# max_features = 20000
# max_seq = 400
# embedding_size = 50
# dropout = 0.25
# lr = 0.0025
# num_workers = 1
# 2020-04-12 22:12:06 ----------------------------------
# 2020-04-12 22:12:06 ---          SUMMARY           ---
# 2020-04-12 22:12:06 ----------------------------------
# 2020-04-12 22:12:06 Total Training Time: 0m 7s
# 2020-04-12 22:12:06 Test Set accuracy: 0.883560%
# 2020-04-12 22:12:06 Total Time: 0m 8s
# 2020-04-12 22:12:06 ----------------------------------
