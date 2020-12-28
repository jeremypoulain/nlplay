import logging
import torch
from torch import nn
from nlplay.data.cache import WordVectorsManager, DSManager, DS, WV
from nlplay.features.text_cleaner import *
from nlplay.models.pytorch.classifiers.custom_rnn import CustomRNN
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
num_epochs = 20
batch_size = 32
ngram_range = (1, 1)
max_features = 20000
max_seq = 150
rnn_type = "gru"
rnn_dropout = 0.2
num_layers = 2
bidirectional = True
embedding_size = 300
hidden_size = 80
lr = 0.00045
update_embedding = True
num_workers = 2

# Data preparation
ds = DSGenerator()
train_ds, val_ds = ds.from_csv(
    train_file=train_csv,
    val_file=test_csv,
    text_col_idx=0,
    label_col_idx=1,
    ngram_range=ngram_range,
    max_features=max_features,
    preprocess_func=base_cleaner,
    preprocess_ncore=3,
    ds_max_seq=max_seq,
)

vecs = get_pretrained_vecs(
    input_vec_file=pretrained_vec,
    target_vocab=ds.vocab,
    dim=embedding_size,
    output_file=None,
)
model = CustomRNN(
    num_classes=ds.num_classes,
    vocabulary_size=ds.vocab_size,
    rnn_type=rnn_type,
    embedding_size=embedding_size,
    rnn_hidden_size=hidden_size,
    rnn_num_layers=num_layers,
    rnn_bidirectional=bidirectional,
    spatial_dropout=rnn_dropout,
    pretrained_vec=vecs,
    update_embedding=update_embedding,
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
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
# num_epochs = 20
# batch_size = 32
# ngram_range = (1, 1)
# max_features = 20000
# max_seq = 150
# rnn_type = "gru"
# rnn_dropout = 0.2
# num_layers = 2
# bidirectional = True
# embedding_size = 300
# hidden_size = 80
# lr = 0.00045
# update_embedding = False
# num_workers = 2
# 2020-12-27 15:50:41 ------------------------------------------
# 2020-12-27 15:50:41 ---              SUMMARY               ---
# 2020-12-27 15:50:41 ------------------------------------------
# 2020-12-27 15:50:41 Number of model parameters : 6301362
# 2020-12-27 15:50:41 Total Training Time: 2m 51s
# 2020-12-27 15:50:41 Total Time: 2m 51s
# 2020-12-27 15:50:41 Best Epoch: 2 - Accuracy Score: 0.882520
# 2020-12-27 15:50:41 ------------------------------------------
