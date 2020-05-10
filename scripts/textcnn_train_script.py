import logging
import torch
from torch import nn
from nlplay.data.cache import WordVectorsManager, WV, DSManager, DS
from nlplay.features.text_cleaner import base_cleaner
from nlplay.models.pytorch.classifiers.textcnn import TextCNN
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
lr = 0.00055
num_workers = 1
kernel_sizes = [100, 100]
filters = [2, 3]

# Data preparation
ds = DSGenerator()
train_ds, val_ds = ds.from_csv(train_file=train_csv, val_file=test_csv, ngram_range=ngram_range,
                               max_features=max_features, preprocess_func=base_cleaner, preprocess_ncore=3,
                               ds_max_seq=max_seq)
vecs = get_pretrained_vecs(input_vec_file=pretrained_vec, target_vocab=ds.vocab,
                           dim=embedding_size, output_file=None)

model = TextCNN(vocabulary_size=len(ds.vocab), num_classes=ds.num_classes,
                model_type='non-static', max_sent_len=max_seq, kernel_sizes=kernel_sizes, filters=filters,
                embedding_dim=embedding_size, pretrained_vec=vecs)

criterion = nn.NLLLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
trainer = PytorchModelTrainer(model, criterion, optimizer,
                              train_ds=train_ds, val_ds=val_ds,
                              batch_size=batch_size, n_workers=num_workers, epochs=num_epochs,)
trainer.train_evaluate()
# 2020-05-09 10:26:47 ----------------------------------
# 2020-05-09 10:26:47 ---          SUMMARY           ---
# 2020-05-09 10:26:47 ----------------------------------
# 2020-05-09 10:26:47 Number of model parameters : 1551002
# 2020-05-09 10:26:47 Total Training Time: 0m 59s
# 2020-05-09 10:26:47 Total Time: 0m 59s
# 2020-05-09 10:26:47 Best Epoch: 7 - Accuracy Score: 0.868640
# 2020-05-09 10:26:47 ----------------------------------