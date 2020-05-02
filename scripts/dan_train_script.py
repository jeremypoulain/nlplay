import logging
import torch
from torch import nn
from nlplay.data.cache import WordVectorsManager, WV, DSManager, DS
from nlplay.features.text_cleaner import *
from nlplay.models.pytorch.classifiers.dan import DAN2L, DAN3L
from nlplay.models.pytorch.dataset import DSGenerator
from nlplay.models.pytorch.pretrained import get_pretrained_vecs
from nlplay.models.pytorch.trainer import PytorchModelTrainer

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG, datefmt="%Y-%m-%d %H:%M:%S")


# Input data files
ds = DSManager(DS.IMDB.value)
train_csv, test_csv, val_csv = ds.get_partition_paths()
lm = WordVectorsManager(WV.GLOVE_EN6B_300.value)
pretrained_vec = lm.get_wv_path()

# Model Parameters
num_epochs = 10
batch_size = 64
ngram_range = (1, 1)
max_features = 25000
max_seq = 600
embedding_size = 300
hidden_size = 256
dropout = 0.3
lr = 0.00025
num_workers = 1

# Data preparation
ds = DSGenerator()
train_ds, val_ds = ds.from_csv(train_file=train_csv, val_file=test_csv, ngram_range=ngram_range,
                               max_features=max_features, preprocess_func=base_cleaner, preprocess_ncore=3,
                               ds_max_seq=max_seq)

vecs = get_pretrained_vecs(input_vec_file=pretrained_vec, target_vocab=ds.vocab,
                           dim=embedding_size, output_file=None,)

model = DAN3L(num_classes=ds.num_classes, vocabulary_size=ds.vocab_size,
              embedding_size=embedding_size, drop_out=dropout,) # pretrained_vec=vecs)

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
trainer = PytorchModelTrainer(model, criterion, optimizer,
                              train_ds=train_ds, val_ds=val_ds,
                              batch_size=batch_size, n_workers=num_workers, epochs=num_epochs)
trainer.train_evaluate()
# num_epochs = 3
# batch_size = 64
# ngram_range = (1, 2)
# max_features = 25000
# max_seq = 600
# embedding_size = 100
# hidden_size = 128
# dropout = 0.1
# lr = 0.00045
# num_workers = 1
# 2020-04-12 23:49:14 ----------------------------------
# 2020-04-12 23:49:14 ---          SUMMARY           ---
# 2020-04-12 23:49:14 ----------------------------------
# 2020-04-12 23:49:14 Total Training Time: 0m 8s
# 2020-04-12 23:49:14 Test Set accuracy: 0.885440%
# 2020-04-12 23:49:14 Total Time: 0m 9s
# 2020-04-12 23:49:14 ----------------------------------