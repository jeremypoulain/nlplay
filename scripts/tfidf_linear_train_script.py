import logging
import torch
from torch import nn
from nlplay.data.cache import DSManager, DS
from nlplay.features.text_cleaner import base_cleaner
from nlplay.models.pytorch.classifiers.linear import SMLinearModel
from nlplay.models.pytorch.dataset import CSRDatasetGenerator
from nlplay.models.pytorch.trainer import PytorchModelTrainer

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG, datefmt="%Y-%m-%d %H:%M:%S")

# Input data files
ds = DSManager(DS.IMDB.value)
train_csv, test_csv, val_csv = ds.get_partition_paths()

# Model Parameters
batch_size = 512
learning_rate = 0.0075
weight_decay = 0.000005
ngram_range = (1, 2)
min_df = 5
max_df = 0.87
max_features = 50000
sublinear_tf = True
stop_words = None
num_epochs = 8
num_workers = 1

# Data preparation
ds = CSRDatasetGenerator()
train_ds, val_ds = ds.from_csv(train_file=train_csv, val_file=test_csv, ngram_range=ngram_range,
                               min_df=min_df, max_df=max_df, max_features=max_features,
                               sublinear_tf=sublinear_tf, stop_words=stop_words,
                               preprocess_func=base_cleaner, preprocess_ncore=3)

model = SMLinearModel(input_size=ds.vocab_size, num_classes=ds.num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Model Training
trainer = PytorchModelTrainer(model, criterion, optimizer,
                              train_ds=train_ds, val_ds=val_ds,
                              batch_size=batch_size, n_workers=num_workers, epochs=num_epochs)

trainer.train_evaluate()
# 2020-04-24 22:09:30 ----------------------------------
# 2020-04-24 22:09:30 ---          SUMMARY           ---
# 2020-04-24 22:09:30 ----------------------------------
# 2020-04-24 22:09:30 Number of model parameters : 100002
# 2020-04-24 22:09:30 Total Training Time: 3m 13s
# 2020-04-24 22:09:30 Total Time: 3m 13s
# 2020-04-24 22:09:30 Best Epoch: 8 - Accuracy Score: 0.902320
# 2020-04-24 22:09:30 ----------------------------------
