import logging
import torch
from torch import nn
from nlplay.features.text_cleaner import base_cleaner
from nlplay.models.pytorch.classifiers.dpcnn import DPCNN
from nlplay.models.pytorch.dataset import DSGenerator
from nlplay.models.pytorch.pretrained import get_pretrained_vecs
from nlplay.models.pytorch.trainer import PytorchModelTrainer

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG, datefmt="%Y-%m-%d %H:%M:%S")

# Model Parameters
train_csv = "../nlplay/data_cache/IMDB/IMDB_train.csv"
test_csv = "../nlplay/data_cache/IMDB/IMDB_test.csv"
pretrained_vec = '../nlplay/data_cache/GLOVE/glove.6B.100d.txt'

num_epochs = 5
batch_size = 128
ngram_range = (1, 1)
max_features = 15000
max_seq = 600
embedding_size = 100
channel_size = 100
dropout = 0.2
lr = 0.00075
num_workers = 1


# Data preparation
ds = DSGenerator()
train_ds, val_ds = ds.from_csv(train_file=train_csv, val_file=test_csv, ngram_range=ngram_range,
                               max_features=max_features, preprocess_func=base_cleaner, preprocess_ncore=3,
                               ds_max_seq=max_seq)
vecs = get_pretrained_vecs(input_vec_file=pretrained_vec, target_vocab=ds.vocab,
                           dim=embedding_size, output_file=None)

model = DPCNN(vocabulary_size=len(ds.vocab), num_classes=ds.num_classes, embedding_size=embedding_size,
              channel_size=channel_size, drop_out=dropout)

criterion = nn.NLLLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
trainer = PytorchModelTrainer(model, criterion, optimizer,
                              train_ds=train_ds, test_ds=val_ds,
                              batch_size=batch_size, n_workers=num_workers, epochs=num_epochs,)
trainer.train()
