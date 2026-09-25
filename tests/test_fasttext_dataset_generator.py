import numpy as np
import pandas as pd
import pytest

from nlplay.features.fasttext_features import FastTextFeaturizer
from nlplay.models.pytorch.classifiers.fasttext_hashed import FastTextDataset
from nlplay.models.pytorch.dataset import FastTextDatasetGenerator

TEXTS = [
    "the movie was great and fun",
    "a terrible boring movie",
    "great acting great story",
    "boring plot and bad acting",
    "fun and great film",
    "bad film awful story",
    "what a great movie",
    "awful and boring",
    "loved the fun story",
    "hated the bad acting",
]
LABELS = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]


@pytest.fixture
def csv_files(tmp_path):
    files = {}
    for name, sl in (("train", slice(0, 10)), ("test", slice(0, 4)), ("val", slice(4, 8))):
        path = tmp_path / f"{name}.csv"
        pd.DataFrame({"text": TEXTS[sl], "label": LABELS[sl]}).to_csv(path, index=False)
        files[name] = str(path)
    return files


def test_features_match_featurizer(csv_files):
    ds = FastTextDatasetGenerator()
    train_ds, val_ds = ds.from_csv(csv_files["train"], val_file=csv_files["val"], word_ngrams=2, bucket=1000)
    featurizer = FastTextFeaturizer(word_ngrams=2, bucket=1000).fit(TEXTS)
    assert isinstance(train_ds, FastTextDataset) and isinstance(val_ds, FastTextDataset)
    assert ds.num_features == featurizer.num_features == train_ds.padding_idx
    for got, expected in zip(train_ds.features, featurizer.transform(TEXTS)):
        assert np.array_equal(got, expected)
    for got, expected in zip(val_ds.features, featurizer.transform(TEXTS[4:8])):
        assert np.array_equal(got, expected)
    assert train_ds.labels.tolist() == LABELS
    assert ds.num_classes == 2 and ds.class_counts == [5, 5]


def test_returned_datasets(csv_files):
    train, test, val = csv_files["train"], csv_files["test"], csv_files["val"]
    assert isinstance(FastTextDatasetGenerator().from_csv(train), FastTextDataset)
    assert len(FastTextDatasetGenerator().from_csv(train, test_file=test)) == 2
    assert len(FastTextDatasetGenerator().from_csv(train, test_file=test, val_file=val)) == 3
    train_ds, val_ds = FastTextDatasetGenerator().from_csv(train, val_size=0.4)
    assert len(train_ds) == 6 and len(val_ds) == 4
    # stratified split, both classes kept in each partition
    assert sorted(val_ds.labels.tolist()) == [0, 0, 1, 1]


def test_preprocess_func_is_applied(csv_files):
    ds = FastTextDatasetGenerator()
    train_ds = ds.from_csv(csv_files["train"], preprocess_func=str.upper, preprocess_ncore=1, word_ngrams=1)
    assert "GREAT" in ds.featurizer.word2id_ and "great" not in ds.featurizer.word2id_
    assert ds.params["preprocess_func"] == "upper"
    assert len(train_ds) == 10
