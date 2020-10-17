import logging
import torch
from torch import nn
from nlplay.data.cache import WordVectorsManager, WV, DSManager, DS
from nlplay.features.text_cleaner import kimyoon_text_cleaner
from nlplay.models.pytorch.classifiers.dpcnn import DPCNN
from nlplay.models.pytorch.dataset import DSGenerator
from nlplay.models.pytorch.pretrained import get_pretrained_vecs
from nlplay.models.pytorch.trainer import PytorchModelTrainer

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG, datefmt="%Y-%m-%d %H:%M:%S")

# Input data files
ds = DSManager(DS.IMDB.value)
train_csv, test_csv, val_csv = ds.get_partition_paths()
lm = WordVectorsManager(WV.GLOVE_EN6B_100.value)
pretrained_vec = lm.get_wv_path()

# Model Parameters
num_epochs = 20
batch_size = 128
ngram_range = (1, 1)
max_features = 15000
max_seq = 600
embedding_size = 100
dropout = 0.2
lr = 0.00075
num_workers = 1


# Data preparation
ds = DSGenerator()
train_ds, val_ds = ds.from_csv(train_file=train_csv, val_file=test_csv, ngram_range=ngram_range,
                               max_features=max_features, preprocess_func=kimyoon_text_cleaner, preprocess_ncore=3,
                               ds_max_seq=max_seq)
vecs = get_pretrained_vecs(input_vec_file=pretrained_vec, target_vocab=ds.vocab,
                           dim=embedding_size, output_file=None)

model = DPCNN(vocabulary_size=len(ds.vocab), num_classes=ds.num_classes, embedding_size=embedding_size,)

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
trainer = PytorchModelTrainer(model, criterion, optimizer,
                              train_ds=train_ds, val_ds=val_ds,
                              batch_size=batch_size, n_workers=num_workers, epochs=num_epochs,)
trainer.train_evaluate()

# 2020-10-17 15:42:29 Epoch: 010/020 | Train Accuracy: 0.994400
# 2020-10-17 15:42:30 Epoch: 010/020 | Val accuracy: 0.824760
# 2020-10-17 15:42:30 Time elapsed: 1m 9s
# 2020-10-17 15:42:30 EarlyStopping patience counter: 3 out of 3
# 2020-10-17 15:42:30 /!\ Early stopping model training /!\
# 2020-10-17 15:42:30 ------------------------------------------
# 2020-10-17 15:42:30 ---              SUMMARY               ---
# 2020-10-17 15:42:30 ------------------------------------------
# 2020-10-17 15:42:30 Number of model parameters : 1509954
# 2020-10-17 15:42:30 Total Training Time: 1m 9s
# 2020-10-17 15:42:30 Total Time: 1m 9s
# 2020-10-17 15:42:30 Best Epoch: 7 - Accuracy Score: 0.826280
# 2020-10-17 15:42:30 ------------------------------------------
