import logging
import pandas as pd
from torch import nn
from nlplay.data.cache import DSManager, DS
from nlplay.features.fasttext_features import FastTextFeaturizer
from nlplay.features.text_cleaner import *
from nlplay.models.pytorch.classifiers.fasttext_hashed import HashedFastText, FastTextDataset, fasttext_sgd
from nlplay.models.pytorch.trainer import PytorchModelTrainer
from nlplay.utils.parlib import parallelApply

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG, datefmt="%Y-%m-%d %H:%M:%S")

# Input data files
ds = DSManager(DS.IMDB.value)
train_csv, test_csv, val_csv = ds.get_partition_paths()

# Inputs & Model Parameters, fastText supervised style: word bigrams, no dropout, SGD with linear decay
num_epochs = 5
batch_size = 32
word_ngrams = 2
minn, maxn = 0, 0   # e.g. 3, 6 to add character n-grams, useful for noisy text / rare words
bucket = 2000000
embedding_size = 50
lr = 0.5
num_workers = 1

# Data preparation
df_train, df_val = pd.read_csv(train_csv), pd.read_csv(test_csv)
train_texts = parallelApply(df_train[df_train.columns[0]], base_cleaner, 3).tolist()
val_texts = parallelApply(df_val[df_val.columns[0]], base_cleaner, 3).tolist()
featurizer = FastTextFeaturizer(word_ngrams=word_ngrams, minn=minn, maxn=maxn, bucket=bucket).fit(train_texts)
train_ds = FastTextDataset(featurizer.transform(train_texts), df_train[df_train.columns[1]].to_numpy(),
                           featurizer.num_features)
val_ds = FastTextDataset(featurizer.transform(val_texts), df_val[df_val.columns[1]].to_numpy(),
                         featurizer.num_features)
num_classes = int(max(train_ds.labels.max(), val_ds.labels.max())) + 1

model = HashedFastText(featurizer.num_features, num_classes, embedding_size=embedding_size, sparse=True)
criterion = nn.CrossEntropyLoss()
optimizer, scheduler = fasttext_sgd(model, lr=lr, total_steps=num_epochs * -(-len(train_ds) // batch_size))

trainer = PytorchModelTrainer(model, criterion, optimizer, lr_scheduler=scheduler,
                              train_ds=train_ds, val_ds=val_ds,
                              batch_size=batch_size, n_workers=num_workers, epochs=num_epochs)
trainer.train_evaluate()
