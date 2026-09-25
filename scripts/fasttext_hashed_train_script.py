import logging
from torch import nn
from nlplay.data.cache import DSManager, DS
from nlplay.features.text_cleaner import *
from nlplay.models.pytorch.classifiers.fasttext_hashed import (
    HashedFastText,
    fasttext_optimizer,
)
from nlplay.models.pytorch.dataset import FastTextDatasetGenerator
from nlplay.models.pytorch.losses import ModifiedHuberLoss
from nlplay.models.pytorch.trainer import PytorchModelTrainer

logging.basicConfig(
    format="%(asctime)s %(message)s", level=logging.DEBUG, datefmt="%Y-%m-%d %H:%M:%S"
)

# Input data files
ds = DSManager(DS.IMDB.value)
train_csv, test_csv, val_csv = ds.get_partition_paths()

# Inputs & Model Parameters
# Tuned on a 20% holdout of the train set. Sparse embeddings, so "adam" trains them with
# SparseAdam. With SGD (fastText style), the minibatch mean loss needs lr = 20, 20 epochs
# for 0.900 on the holdout vs 0.911 with Adam.
num_epochs = 5
batch_size = 32
word_ngrams = 2
minn, maxn = 0, 0  # e.g. 3, 6 for character n-grams, lower and 4x slower on IMDB
bucket = 2000000
embedding_size = 50
loss = "softmax"  # "hs" for the hierarchical softmax, worth it with many classes
criterion_name = "cross_entropy"  # or "modified_huber", same accuracy with lr = 0.003
optimizer_name = "adam"
lr = 0.01
pretrained_vec_file = None  # e.g. a fastText .vec file of dimension embedding_size
num_workers = 1

# Data preparation
ds = FastTextDatasetGenerator()
train_ds, val_ds = ds.from_csv(
    train_file=train_csv,
    val_file=test_csv,
    preprocess_func=base_cleaner,
    preprocess_ncore=3,
    word_ngrams=word_ngrams,
    minn=minn,
    maxn=maxn,
    bucket=bucket,
    pretrained_vec_file=pretrained_vec_file,
)

model = HashedFastText(
    ds.num_features,
    ds.num_classes,
    embedding_size=embedding_size,
    sparse=True,
    loss=loss,
    class_counts=ds.class_counts,
)
if pretrained_vec_file:
    model.load_word_vectors(ds.featurizer, ds.pretrained_words, ds.pretrained_vectors)

# with loss="hs" the trainer uses model.training_loss, the criterion is not used for training
if loss == "hs":
    criterion = nn.NLLLoss()
elif criterion_name == "modified_huber":
    criterion = ModifiedHuberLoss()
else:
    criterion = nn.CrossEntropyLoss()
optimizer, scheduler = fasttext_optimizer(
    model,
    lr=lr,
    total_steps=num_epochs * -(-len(train_ds) // batch_size),
    optimizer=optimizer_name,
)

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

# num_epochs = 5
# batch_size = 32
# word_ngrams = 2
# minn, maxn = 0, 0
# bucket = 2000000
# embedding_size = 50
# loss = "softmax"
# criterion_name = "cross_entropy"
# optimizer_name = "adam"
# lr = 0.01
# num_workers = 1
# 2026-09-25 08:19:51 ------------------------------------------
# 2026-09-25 08:19:51 ---              SUMMARY               ---
# 2026-09-25 08:19:51 ------------------------------------------
# 2026-09-25 08:19:51 Number of model parameters : 103907100
# 2026-09-25 08:19:51 Total Training Time: 0m 13s
# 2026-09-25 08:19:51 Total Time: 0m 13s
# 2026-09-25 08:19:51 Best Epoch: 1 - Accuracy Score: 0.905480
# 2026-09-25 08:19:51 ------------------------------------------
# Early stop after epoch 4, test accuracy 0.899 to 0.901 after epoch 1, train accuracy 1.0
