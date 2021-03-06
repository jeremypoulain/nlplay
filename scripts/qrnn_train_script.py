import logging
import torch
from torch import nn
from nlplay.data.cache import WordVectorsManager, WV, DS, DSManager
from nlplay.models.pytorch.classifiers.qrnn import QRNN
from nlplay.models.pytorch.pretrained import get_pretrained_vecs
from nlplay.features.text_cleaner import *
from nlplay.models.pytorch.trainer import PytorchModelTrainer
from nlplay.models.pytorch.dataset import DSGenerator
from nlplay.utils import utils


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG, datefmt="%Y-%m-%d %H:%M:%S")

# Input data files
ds = DSManager(DS.IMDB.value)
train_csv, test_csv, val_csv = ds.get_partition_paths()
lm = WordVectorsManager(WV.GLOVE_EN6B_100.value)
pretrained_vec = lm.get_wv_path()


# Model Parameters
num_epochs = 10
batch_size = 64
ngram_range = (1, 1)
max_features = 20000
max_seq = 80
embedding_size = 100
dropout = 0.3
lr = 0.001
num_workers = 1

# Data preparation
ds = DSGenerator()
train_ds, val_ds = ds.from_csv(train_file=train_csv, val_file=test_csv, ngram_range=ngram_range,
                               max_features=max_features, preprocess_func=base_cleaner, preprocess_ncore=3,
                               ds_max_seq=max_seq)
vecs = get_pretrained_vecs(input_vec_file=pretrained_vec, target_vocab=ds.vocab,
                           dim=embedding_size, output_file=None)

# Model
model = QRNN(num_classes=ds.num_classes, vocabulary_size=ds.vocab_size,
             embedding_size=embedding_size, drop_out=dropout)

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
trainer = PytorchModelTrainer(model, criterion, optimizer,
                              train_ds=train_ds, val_ds=val_ds,
                              batch_size=batch_size, n_workers=num_workers, epochs=num_epochs)
trainer.train_evaluate()
# 2020-01-12 16:19:23 ----------------------------------
# 2020-01-12 16:19:23 ---          SUMMARY           ---
# 2020-01-12 16:19:23 ----------------------------------
# 2020-01-12 16:19:23 Total Training Time: 6m 15s
# 2020-01-12 16:19:29 Test Set accuracy: 0.835640%
# 2020-01-12 16:19:29 Total Time: 6m 21s
# 2020-01-12 16:19:29 ----------------------------------