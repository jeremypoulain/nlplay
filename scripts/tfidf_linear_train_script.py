import logging
from pathlib import Path
import torch
from torch import nn

from nlplay.features.text_cleaner import base_cleaner
from nlplay.models.pytorch.classifiers.linear import LinearModel
from nlplay.models.pytorch.dataset import CSRDatasetGenerator
from nlplay.models.pytorch.trainer import PytorchModelTrainer

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG, datefmt="%Y-%m-%d %H:%M:%S")

# Model Parameters
train_csv = "../nlplay/data_cache/IMDB/IMDB_train.csv"
test_csv = "../nlplay/data_cache/IMDB/IMDB_test.csv"
batch_size = 512
learning_rate = 0.0075
weight_decay = 0.000005
num_epochs = 8
num_workers = 1

# Data preparation
ds = CSRDatasetGenerator()
train_ds, test_ds = ds.from_csv(train_file=train_csv, test_file=test_csv, ngram_range=(1, 2),
                                min_df=5, max_df=0.87, max_features=50000,
                                sublinear_tf=True, stop_words=None,
                                preprocess_func=base_cleaner, preprocess_ncore=3)

model = LinearModel(input_size=ds.vocab_size, num_classes=ds.num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Model Training
trainer = PytorchModelTrainer(model, criterion, optimizer,
                              train_ds=train_ds, test_ds=test_ds,
                              batch_size=batch_size, n_workers=num_workers, epochs=num_epochs)

trainer.train()
# 2020-04-24 22:09:30 ----------------------------------
# 2020-04-24 22:09:30 ---          SUMMARY           ---
# 2020-04-24 22:09:30 ----------------------------------
# 2020-04-24 22:09:30 Number of model parameters : 100002
# 2020-04-24 22:09:30 Total Training Time: 3m 13s
# 2020-04-24 22:09:30 Total Time: 3m 13s
# 2020-04-24 22:09:30 Best Epoch: 8 - Accuracy Score: 0.902320
# 2020-04-24 22:09:30 ----------------------------------
