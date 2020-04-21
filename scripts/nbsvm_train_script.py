import logging
from pathlib import Path
import torch
from torch import nn
from nlplay.features.text_cleaner import ft_cleaner, base_cleaner
from nlplay.models.pytorch.classifiers.nbsvm import NBSVM
from nlplay.models.pytorch.dataset import NBSVMDatasetGenerator
from nlplay.models.pytorch.trainer import PytorchModelTrainer
from nlplay.utils import utils

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG, datefmt="%Y-%m-%d %H:%M:%S")

# Model Parameters
train_csv = "../nlplay/data_cache/IMDB/IMDB_train.csv"
test_csv = "../nlplay/data_cache/IMDB/IMDB_test.csv"
num_epochs = 3
batch_size = 256
ngram_range = (1, 3)
max_features = 200000
lr = 0.02
weight_decay = 1e-6
ds_max_seq = 2000
num_workers = 1


# Data preparation
ds = NBSVMDatasetGenerator()
r, train_ds, test_ds = ds.from_csv(train_file=train_csv, test_file=test_csv, ngram_range=ngram_range,
                                   max_features=max_features, stop_words=None,
                                   preprocess_func=base_cleaner, preprocess_ncore=3, ds_max_seq=ds_max_seq)
# ds.to_numpy(input_folder + "/IMDB/")
# r, train_ds, train_dl, test_ds, test_dl = ds.from_numpy(train_data_file=input_folder + "/IMDB/train_data_nbsvm.npz",
#                                                         val_data_file=input_folder + "/IMDB/test_data_nbsvm.npz")

# Model
model = NBSVM(ds.vocab_size, ds.num_classes, r)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
trainer = PytorchModelTrainer(model, criterion, optimizer,
                              train_ds=train_ds, test_ds=test_ds,
                              batch_size=batch_size, n_workers=num_workers, epochs=num_epochs)
trainer.train()
# num_epochs = 3
# batch_size = 256
# ngram_range = (1, 3)
# max_features = 200000
# lr = 0.02
# weight_decay = 1e-6
# ds_max_seq = 2000
# num_workers = 1
# 2020-04-13 13:48:30 ----------------------------------
# 2020-04-13 13:48:30 ---          SUMMARY           ---
# 2020-04-13 13:48:30 ----------------------------------
# 2020-04-13 13:48:30 Total Training Time: 0m 5s
# 2020-04-13 13:48:30 Test Set accuracy: 0.922280%
# 2020-04-13 13:48:30 Total Time: 0m 6s
# 2020-04-13 13:48:30 ----------------------------------