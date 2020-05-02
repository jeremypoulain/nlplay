import logging
import torch
from torch import nn
from nlplay.data.cache import DSManager, DS
from nlplay.features.text_cleaner import base_cleaner
from nlplay.models.pytorch.classifiers.nbsvm import NBSVM
from nlplay.models.pytorch.dataset import NBSVMDatasetGenerator
from nlplay.models.pytorch.trainer import PytorchModelTrainer

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG, datefmt="%Y-%m-%d %H:%M:%S")

# Input data files
ds = DSManager(DS.IMDB.value)
train_csv, test_csv, val_csv = ds.get_partition_paths()

# Model Parameters
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
r, train_ds, val_ds = ds.from_csv(train_file=train_csv, val_file=test_csv, ngram_range=ngram_range,
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
                              train_ds=train_ds, val_ds=val_ds,
                              batch_size=batch_size, n_workers=num_workers, epochs=num_epochs)
trainer.train_evaluate()
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