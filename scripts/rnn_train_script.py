import logging
import torch
from torch import nn
from nlplay.data.cache import WordVectorsManager, DSManager, DS, WV
from nlplay.features.text_cleaner import *
from nlplay.models.pytorch.classifiers.rnn import RNN
from nlplay.models.pytorch.dataset import DSGenerator
from nlplay.models.pytorch.pretrained import get_pretrained_vecs
from nlplay.models.pytorch.trainer import PytorchModelTrainer

logging.basicConfig(
    format="%(asctime)s %(message)s", level=logging.DEBUG, datefmt="%Y-%m-%d %H:%M:%S"
)

# Input data files
ds = DSManager(DS.IMDB.value)
train_csv, test_csv, val_csv = ds.get_partition_paths()
lm = WordVectorsManager(WV.GLOVE_EN6B_300.value)
pretrained_vec = lm.get_wv_path()

# Model Parameters
num_epochs = 40
batch_size = 64
ngram_range = (1, 1)
max_features = 20000
max_seq = 200
rnn_type = "GRU"
rnn_dropout = 0.25
num_layers = 2
bidirectionnal = True
embedding_size = 300
hidden_size = 128
lr = 0.0015
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

vecs = get_pretrained_vecs(input_vec_file=pretrained_vec, target_vocab=ds.vocab,
                           dim=embedding_size, output_file=None)
model = RNN(
    num_classes=ds.num_classes,
    vocabulary_size=ds.vocab_size,
    rnn_type=rnn_type,
    embedding_size=embedding_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    dropout=rnn_dropout,
    pretrained_vec=vecs,
    update_embedding=False
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
trainer = PytorchModelTrainer(
    model,
    criterion,
    optimizer,
    train_ds=train_ds,
    val_ds=val_ds,
    batch_size=batch_size,
    n_workers=num_workers,
    epochs=num_epochs,
)
trainer.train_evaluate()
# Model Parameters
# num_epochs = 40
# batch_size = 64
# ngram_range = (1, 1)
# max_features = 20000
# max_seq = 200
# rnn_type = "gru"
# rnn_dropout = 0.25
# num_layers = 2
# bidirectionnal = True
# embedding_size = 300
# hidden_size = 128
# lr = 0.0015
# num_workers = 1
# 2020-05-02 14:54:14 ----------------------------------
# 2020-05-02 14:54:14 ---          SUMMARY           ---
# 2020-05-02 14:54:14 ----------------------------------
# 2020-05-02 14:54:14 Number of model parameters : 6265650
# 2020-05-02 14:54:14 Total Training Time: 1m 25s
# 2020-05-02 14:54:14 Total Time: 1m 25s
# 2020-05-02 14:54:14 Best Epoch: 5 - Accuracy Score: 0.861160
# 2020-05-02 14:54:14 ----------------------------------