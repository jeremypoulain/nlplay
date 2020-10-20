import logging
import torch
from torch import nn
from nlplay.data.cache import DSManager, DS, WordVectorsManager, WV
from nlplay.features.text_cleaner import *
from nlplay.models.pytorch.classifiers.rcnn import TextRCNN
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

# Inputs & Model Parameters
num_epochs = 4
batch_size = 64
ngram_range = (1, 1)
max_seq = 150
max_features = 20000
embedding_size = 300
rnn_type = "lstm"
activation_function = "tanh"
hidden_size = 100
rnn_dropout = 0.1
dropout = 0.2
num_layers = 2
bidirectional = True
update_embedding = False
lr = 0.00075
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

vecs = get_pretrained_vecs(
    input_vec_file=pretrained_vec,
    target_vocab=ds.vocab,
    dim=embedding_size,
    output_file=None,
)

model = TextRCNN(
    num_classes=ds.num_classes,
    vocabulary_size=ds.vocab_size,
    embedding_size=embedding_size,
    activation_function=activation_function,
    hidden_size=hidden_size,
    rnn_type=rnn_type,
    rnn_num_layers=num_layers,
    rnn_bidirectional=bidirectional,
    rnn_dropout=rnn_dropout,
    drop_out=dropout,
    update_embedding=update_embedding
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
