import gc
import logging
import os
import re
import string
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras_preprocessing.text import Tokenizer
from keras_preprocessing import sequence, text
from nlplay.features.text_vectorizer import DataVectorizer
from nlplay.utils.parlib import parallelApply
from nlplay.utils.utils import get_elapsed_time


class CSRDataset(Dataset):
    """Custom dataset for PyTorch"""

    def __init__(self, csr_matrix, labels):
        """Initialization"""
        self.csr_matrix = csr_matrix
        self.labels = labels

    def __len__(self):
        """Get the total number of samples"""
        return self.csr_matrix.shape[0]

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Load/Convert vectorized data and get label
        X = torch.from_numpy(self.csr_matrix[index].toarray().squeeze()).float()
        y = int(self.labels[index])

        return X, y


class CSRDatasetGenerator(object):
    def __init__(self, seed: int = 123):

        self.seed = seed
        self.train_file = None
        self.test_file = None
        self.val_file = None

        self.text_col_idx = None
        self.label_col_idx = None

        self.vocab_size = None
        self.num_classes = None

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_val = None
        self.y_val = None

        self.vectorizer = None
        logging.getLogger(__name__)

    def from_csv(
        self,
        train_file: str,
        test_file: str = None,
        val_file: str = None,
        val_size: float = 0.0,
        text_col_idx=0,
        label_col_idx=1,
        sep: str = ",",
        header=0,
        encoding: str = "utf8",
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.9,
        use_idf: bool = True,
        sublinear_tf: bool = False,
        norm="l2",
        binary=False,
        max_features=None,
        stop_words=None,
        preprocess_func=None,
        preprocess_ncore=2,
    ):

        logging.info("Starting Data Preparation...")
        start_time = time.time()

        self.train_file = train_file
        self.test_file = test_file
        self.val_file = val_file

        self.text_col_idx = text_col_idx
        self.label_col_idx = label_col_idx

        re_tok = re.compile("([%s“”¨«»®´·º½¾¿¡§£₤‘’])" % string.punctuation)
        tokenizer = lambda x: re_tok.sub(r" \1 ", x).split()

        self.vectorizer = TfidfVectorizer(
            use_idf=use_idf,
            tokenizer=tokenizer,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            norm=norm,
            binary=binary,
            max_features=max_features,
            stop_words=stop_words,
        )

        df = pd.read_csv(self.train_file, sep=sep, encoding=encoding, header=header)
        if preprocess_func is not None:
            df[df.columns[self.text_col_idx]] = parallelApply(
                df[df.columns[self.text_col_idx]], preprocess_func, preprocess_ncore
            )
        X = df[df.columns[self.text_col_idx]].tolist()
        y = df[df.columns[self.label_col_idx]].to_numpy(float)
        del df

        self.X_train = self.vectorizer.fit_transform(X)
        self.y_train = y
        self.vocab_size = len([v for k, v in self.vectorizer.vocabulary_.items()])
        self.num_classes = len(np.unique(self.y_train))
        del X, y

        train_ds = CSRDataset(self.X_train, self.y_train)
        gc.collect()

        if self.test_file is not None:
            df = pd.read_csv(self.test_file, sep=sep, encoding=encoding, header=header)
            if preprocess_func is not None:
                df[df.columns[self.text_col_idx]] = parallelApply(
                    df[df.columns[self.text_col_idx]], preprocess_func, preprocess_ncore
                )
            X = df[df.columns[self.text_col_idx]].tolist()
            y = df[df.columns[self.label_col_idx]].to_numpy(float)
            del df
            self.X_test = self.vectorizer.transform(X)
            self.y_test = y
            del X, y
            test_ds = CSRDataset(self.X_test, self.y_test)
            gc.collect()

        if self.val_file is not None:  # or val_size > 0.0:
            df = pd.read_csv(self.val_file, sep=sep, encoding=encoding, header=header)
            if preprocess_func is not None:
                df[df.columns[self.text_col_idx]] = parallelApply(
                    df[df.columns[self.text_col_idx]], preprocess_func, preprocess_ncore
                )
            X = df[df.columns[self.text_col_idx]].tolist()
            y = df[df.columns[self.label_col_idx]].to_numpy(float)
            del df
            self.X_val = self.vectorizer.transform(X)
            self.y_val = y
            del X, y
            val_ds = CSRDataset(self.X_val, self.y_val)

            gc.collect()

        logging.info(
            "Data Preparation Completed - Time elapsed: " + get_elapsed_time(start_time)
        )

        if self.val_file is not None:
            if self.test_file is not None:
                return train_ds, test_ds, val_ds
            else:
                return train_ds, val_ds
        else:
            return train_ds

    def from_numpy(
        self,
        train_data_file: str,
        test_data_file: str = None,
        val_data_file: str = None,
    ):

        logging.info("Starting Data Preparation...")
        start_time = time.time()

        train_npz = np.load(train_data_file, allow_pickle=True)
        self.X_train = train_npz["X"].item()
        self.y_train = train_npz["y"]

        self.num_classes = len(np.unique(self.y_train))
        self.vocab_size = np.shape(self.X_train)[1]

        train_ds = CSRDataset(self.X_train, self.y_train)

        if test_data_file is not None:
            test_npz = np.load(test_data_file, allow_pickle=True)
            self.X_test = test_npz["X"].item()
            self.y_test = test_npz["y"]

            test_ds = CSRDataset(self.X_test, self.y_test)

        if val_data_file is not None:
            val_npz = np.load(val_data_file, allow_pickle=True)
            self.X_val = val_npz["X"].item()
            self.y_val = val_npz["y"]

            val_ds = CSRDataset(self.X_val, self.y_val)

        logging.info(
            "Data Import Completed - Time elapsed: %.2f min"
            % ((time.time() - start_time) / 60)
        )

        if val_data_file is not None:
            if test_data_file is not None:
                return train_ds, val_ds, test_ds
            else:
                return train_ds, val_ds
        else:
            return train_ds

    def to_numpy(self, dest_folder: str):

        logging.info("Starting Data Export...")
        start_time = time.time()

        np.savez_compressed(
            os.path.join(dest_folder, "train_data.npz"), X=self.X_train, y=self.y_train
        )

        if self.X_test is not None:
            np.savez_compressed(
                os.path.join(dest_folder, "test_data.npz"), X=self.X_test, y=self.y_test
            )

        if self.X_val is not None:
            np.savez_compressed(
                os.path.join(dest_folder, "val_data.npz"), X=self.X_val, y=self.y_val
            )

        logging.info(
            "Data Export Completed - Time elapsed: " + get_elapsed_time(start_time)
        )


class NBSVMDatasetGenerator(object):
    def __init__(self, seed: int = 123):

        self.seed = seed
        self.train_file = None
        self.test_file = None
        self.val_file = None

        self.text_col_idx = None
        self.label_col_idx = None

        self.vocab_size = None
        self.num_classes = None

        self.X_train = None
        self.X_train_words_seq = None
        self.y_train = None
        self.X_test = None
        self.X_test_words_seq = None
        self.y_test = None
        self.X_val = None
        self.X_val_words_seq = None
        self.y_val = None

        self.r = None
        self.vectorizer = None

        logging.getLogger(__name__)

    def calc_r(self, y_i, x, y):
        x = x.sign()
        p = x[np.argwhere(y == y_i)[:, 0]].sum(axis=0) + 1
        q = x[np.argwhere(y != y_i)[:, 0]].sum(axis=0) + 1
        p, q = np.asarray(p).squeeze(), np.asarray(q).squeeze()
        return np.log((p / p.sum()) / (q / q.sum()))

    def _bow2adjlist(self, X, max_seq=None):
        x = coo_matrix(X)
        _, counts = np.unique(x.row, return_counts=True)
        pos = np.hstack([np.arange(c) for c in counts])
        adjlist = csr_matrix((x.col + 1, (x.row, pos)))
        datlist = csr_matrix((x.data, (x.row, pos)))

        if max_seq is not None:
            adjlist, datlist = adjlist[:, :max_seq], datlist[:, :max_seq]

        return adjlist, datlist

    def from_csv(
        self,
        train_file: str,
        test_file: str = None,
        val_file: str = None,
        val_size: float = 0.0,
        text_col_idx=0,
        label_col_idx=1,
        sep: str = ",",
        header=0,
        encoding: str = "utf8",
        ngram_range=(1, 3),
        min_df=1,
        max_df=1.0,
        use_idf: bool = False,
        sublinear_tf: bool = False,
        norm="l2",
        binary=False,
        max_features=None,
        stop_words=None,
        preprocess_func=None,
        preprocess_ncore=2,
        ds_max_seq=1000,
        ds_type="Dataset",
    ):

        logging.info("Starting Data Preparation...")
        logging.info("  Training Data ...")
        start_time = time.time()

        self.train_file = train_file
        self.test_file = test_file
        self.val_file = val_file

        self.text_col_idx = text_col_idx
        self.label_col_idx = label_col_idx

        re_tok = re.compile("([%s“”¨«»®´·º½¾¿¡§£₤‘’])" % string.punctuation)
        tokenizer = lambda x: re_tok.sub(r" \1 ", x).split()
        #
        self.vectorizer = TfidfVectorizer(
            use_idf=use_idf,
            tokenizer=tokenizer,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            norm=norm,
            binary=binary,
            max_features=max_features,
            stop_words=stop_words,
        )

        df = pd.read_csv(self.train_file, sep=sep, encoding=encoding, header=header)
        if preprocess_func is not None:
            df[df.columns[self.text_col_idx]] = parallelApply(
                df[df.columns[self.text_col_idx]], preprocess_func, preprocess_ncore
            )
        X = df[df.columns[self.text_col_idx]].tolist()
        y = df[df.columns[self.label_col_idx]].to_numpy(int)
        del df
        self.X_train = self.vectorizer.fit_transform(X)
        self.y_train = y
        self.vocab_size = len([v for k, v in self.vectorizer.vocabulary_.items()])
        self.num_classes = len(np.unique(self.y_train))
        self.X_train_words_seq, _ = self._bow2adjlist(self.X_train, max_seq=ds_max_seq)
        self.r = np.column_stack(
            [
                self.calc_r(i, self.X_train, self.y_train)
                for i in range(self.num_classes)
            ]
        )
        del X, y

        train_ds = TensorDataset(
            torch.from_numpy(self.X_train_words_seq.toarray()).long(),
            torch.from_numpy(self.y_train).long(),
        )
        gc.collect()

        if self.test_file is not None:
            logging.info("  Test Data ...")
            df = pd.read_csv(self.test_file, sep=sep, encoding=encoding, header=header)
            if preprocess_func is not None:
                df[df.columns[self.text_col_idx]] = parallelApply(
                    df[df.columns[self.text_col_idx]], preprocess_func, preprocess_ncore
                )
            X = df[df.columns[self.text_col_idx]].tolist()
            y = df[df.columns[self.label_col_idx]].to_numpy(int)
            del df
            self.X_test = self.vectorizer.transform(X)
            self.y_test = y
            self.X_test_words_seq, _ = self._bow2adjlist(
                self.X_test, max_seq=ds_max_seq
            )
            del X, y
            test_ds = TensorDataset(
                torch.from_numpy(self.X_test_words_seq.toarray()).long(),
                torch.from_numpy(self.y_test).long(),
            )
            gc.collect()

        if self.val_file is not None:
            logging.info("  Validation Data ...")
            df = pd.read_csv(self.val_file, sep=sep, encoding=encoding, header=header)
            if preprocess_func is not None:
                df[df.columns[self.text_col_idx]] = parallelApply(
                    df[df.columns[self.text_col_idx]], preprocess_func, preprocess_ncore
                )
            X = df[df.columns[self.text_col_idx]].tolist()
            y = df[df.columns[self.label_col_idx]].to_numpy(int)
            del df
            self.X_val = self.vectorizer.transform(X)
            self.y_val = y
            self.X_val_words_seq, _ = self._bow2adjlist(self.X_val, max_seq=ds_max_seq)
            del X, y
            val_ds = TensorDataset(
                torch.from_numpy(self.X_val_words_seq.toarray()).long(),
                torch.from_numpy(self.y_val).long(),
            )
            gc.collect()

        logging.info(
            "Data Preparation Completed - Time elapsed: " + get_elapsed_time(start_time)
        )
        if self.val_file is not None:
            if self.test_file is not None:
                return self.r, train_ds, test_ds, val_ds
            else:
                return self.r, train_ds, val_ds
        else:
            return self.r, train_ds

    def from_numpy(
        self,
        train_data_file: str,
        test_data_file: str = None,
        val_data_file: str = None,
        ds_type="Dataset",
    ):

        logging.info("Starting Data Preparation...")
        start_time = time.time()

        train_npz = np.load(train_data_file, allow_pickle=True)
        self.X_train = train_npz["X"].item()
        self.y_train = train_npz["y"]

        self.num_classes = len(np.unique(self.y_train))
        self.vocab_size = np.shape(self.X_train)[1]

        train_ds = CSRDataset(self.X_train, self.y_train)

        if test_data_file is not None:
            test_npz = np.load(test_data_file, allow_pickle=True)
            self.X_test = test_npz["X"].item()
            self.y_test = test_npz["y"]

            test_ds = CSRDataset(self.X_test, self.y_test)

        if val_data_file is not None:
            val_npz = np.load(val_data_file, allow_pickle=True)
            self.X_val = val_npz["X"].item()
            self.y_val = val_npz["y"]

            val_ds = CSRDataset(self.X_val, self.y_val)

        logging.info(
            "Data Import Completed - Time elapsed: " + get_elapsed_time(start_time)
        )

        if self.val_file is not None:
            if self.test_file is not None:
                return self.r, train_ds, test_ds, val_ds
            else:
                return self.r, train_ds, val_ds
        else:
            return self.r, train_ds

    def to_numpy(self, dest_folder: str):

        logging.info("Starting Data Export...")
        start_time = time.time()

        if self.X_train is not None:
            np.savez_compressed(
                os.path.join(dest_folder, "train_data_nbsvm.npz"),
                X=self.X_train,
                y=self.y_train,
            )

        if self.X_test is not None:
            np.savez_compressed(
                os.path.join(dest_folder, "test_data_nbsvm.npz"),
                X=self.X_test,
                y=self.y_test,
            )

        if self.X_val is not None:
            np.savez_compressed(
                os.path.join(dest_folder, "val_data_nbsvm.npz"),
                X=self.X_val,
                y=self.y_val,
            )

        logging.info(
            "Data Export Completed - Time elapsed: " + get_elapsed_time(start_time)
        )


class NNDataset(Dataset):
    """Custom dataset for PyTorch"""

    def __init__(self, X_data, Y_data, max_seq=400, replace_unk=False):
        """Initialization"""
        self.X_data = X_data
        self.Y_data = Y_data
        self.max_seq = max_seq
        self.replace_unk = replace_unk

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.X_data)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Load/Convert vectorized data and get label
        record = np.asarray(self.X_data[index], dtype=int)
        record.resize((1, self.max_seq), refcheck=False)
        if self.replace_unk:
            record = np.where(record == 1, 0, record)
        X = torch.from_numpy(record.squeeze())
        y = int(self.Y_data[index])

        return X, y


class NNDatasetGenerator(object):
    def __init__(self, seed: int = 123):

        self.seed = seed
        self.train_file = None
        self.test_file = None
        self.val_file = None

        self.text_col_idx = None
        self.label_col_idx = None
        self.tokenizer = None
        self.vocab = None
        self.vocab_size = None
        self.num_classes = None

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_val = None
        self.y_val = None

        logging.basicConfig(
            format="%(asctime)s %(message)s",
            level=logging.DEBUG,
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def from_csv(
        self,
        train_file: str,
        test_file: str = None,
        val_file: str = None,
        val_size: float = 0.1,
        text_col_idx=0,
        label_col_idx=1,
        sep: str = ",",
        header=0,
        encoding: str = "utf8",
        preprocess_func=None,
        preprocess_ncore=2,
        ngram_range=(1, 3),
        max_features=20000,
        ds_max_seq=1000,
        ds_type="TensorDataset",
    ):

        logging.info("Starting Data Preparation...")
        start_time = time.time()

        self.train_file = train_file
        self.test_file = test_file
        self.val_file = val_file

        self.text_col_idx = text_col_idx
        self.label_col_idx = label_col_idx

        df = pd.read_csv(self.train_file, sep=sep, encoding=encoding, header=header)
        if preprocess_func is not None:
            df[df.columns[self.text_col_idx]] = parallelApply(
                df[df.columns[self.text_col_idx]], preprocess_func, preprocess_ncore
            )
        X = df[df.columns[self.text_col_idx]].tolist()
        y = df[df.columns[self.label_col_idx]].to_numpy(dtype=int)

        del df
        logging.info("Adding 1-gram features".format(ngram_range[1]))
        self.tokenizer = Tokenizer(num_words=max_features, lower=False, filters="")
        self.tokenizer.fit_on_texts(X)
        self.X_train = self.tokenizer.texts_to_sequences(X)

        if ngram_range[1] > 1:
            logging.info("Adding N-gram features".format(ngram_range[1]))
            # Create set of unique n-gram from the training set.
            ngram_set = set()
            for input_list in self.X_train:
                for i in range(2, ngram_range[1] + 1):
                    set_of_ngram = self.create_ngram_set(input_list, ngram_value=i)
                    ngram_set.update(set_of_ngram)

            # Dictionary mapping n-gram token to a unique integer.
            # Integer values are greater than max_features in order
            # to avoid collision with existing features.
            start_index = max_features + 1
            token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
            indice_token = {token_indice[k]: k for k in token_indice}

            # max_features is the highest integer that could be found in the dataset.
            max_features = np.max(list(indice_token.keys())) + 1

            # Augmenting input tokens with n-grams features
            self.X_train = self.add_ngram(self.X_train, token_indice, ngram_range[1])

        self.X_train = sequence.pad_sequences(self.X_train, maxlen=ds_max_seq)
        self.y_train = y

        self.vocab_size = max_features
        logging.info("Building final vocab...")
        vocab_wrd_idx = set()
        _ = [vocab_wrd_idx.add(idx) for sent in self.X_train for idx in sent]
        del _
        self.vocab = {
            self.tokenizer.index_word[i]: i
            for i in vocab_wrd_idx
            if i in self.tokenizer.index_word
        }
        # self.strt = start_index
        # if ngram_range[1] > 1:
        #     self._start = start_index
        #     a = [str(indice_token[i]) for i in range(start_index, len(vocab_wrd_idx)) if i in indice_token[i]]
        self.num_classes = len(np.unique(self.y_train))
        del X, y
        gc.collect()

        if ds_type == "TensorDataset":
            train_ds = TensorDataset(
                torch.from_numpy(self.X_train).long(),
                torch.from_numpy(self.y_train).long(),
            )
        else:
            train_ds = NNDataset(self.X_train, self.y_train, max_seq=ds_max_seq)

        if self.test_file is not None:
            df = pd.read_csv(self.test_file, sep=sep, encoding=encoding, header=header)
            if preprocess_func is not None:
                df[df.columns[self.text_col_idx]] = parallelApply(
                    df[df.columns[self.text_col_idx]], preprocess_func, preprocess_ncore
                )
            X = df[df.columns[self.text_col_idx]].tolist()
            y = df[df.columns[self.label_col_idx]].to_numpy(dtype=int)
            del df
            self.X_test = self.tokenizer.texts_to_sequences(X)
            if ngram_range[1] > 1:
                self.X_test = self.add_ngram(self.X_test, token_indice, ngram_range[1])
            self.X_test = sequence.pad_sequences(self.X_train, maxlen=ds_max_seq)
            self.y_test = y
            del X, y
            gc.collect()
            if ds_type == "TensorDataset":
                test_ds = TensorDataset(
                    torch.from_numpy(self.X_test).long(),
                    torch.from_numpy(self.y_test).long(),
                )
            else:
                test_ds = NNDataset(self.X_test, self.y_test, max_seq=ds_max_seq)

        if self.val_file is not None:
            df = pd.read_csv(self.val_file, sep=sep, encoding=encoding)
            if preprocess_func is not None:
                df[df.columns[self.text_col_idx]] = parallelApply(
                    df[df.columns[self.text_col_idx]], preprocess_func, preprocess_ncore
                )
            X = df[df.columns[self.text_col_idx]].tolist()
            y = df[df.columns[self.label_col_idx]].to_numpy(dtype=int)
            del df
            self.X_val = self.tokenizer.texts_to_sequences(X)
            if ngram_range[1] > 1:
                self.X_val = self.add_ngram(self.X_val, token_indice, ngram_range[1])
            self.X_val = sequence.pad_sequences(self.X_val, maxlen=ds_max_seq)
            self.y_val = y
            del X, y
            gc.collect()

            if ds_type == "TensorDataset":
                val_ds = TensorDataset(
                    torch.from_numpy(self.X_val).long(),
                    torch.from_numpy(self.y_val).long(),
                )
            else:
                val_ds = NNDataset(self.X_val, self.y_val, max_seq=ds_max_seq)

        logging.info(
            "Data Preparation Completed - Time elapsed: " + get_elapsed_time(start_time)
        )

        if self.val_file is not None:
            if self.test_file is not None:
                return train_ds, test_ds, val_ds
            else:
                return train_ds, val_ds
        else:
            return train_ds

    def create_ngram_set(self, input_list, ngram_value=2):
        return set(zip(*[input_list[i:] for i in range(ngram_value)]))

    def add_ngram(self, sequences, token_indice, ngram_range=2):
        new_sequences = []
        for input_list in sequences:
            new_list = input_list[:]
            for ngram_value in range(2, ngram_range + 1):
                for i in range(len(new_list) - ngram_value + 1):
                    ngram = tuple(new_list[i : i + ngram_value])
                    if ngram in token_indice:
                        new_list.append(token_indice[ngram])
            new_sequences.append(new_list)

        return new_sequences

    def to_numpy(self, dest_folder: str):

        logging.info("Starting Data Export...")
        start_time = time.time()

        if self.tokenizer is not None:
            self.tokenizer.to_json(os.path.join(dest_folder, "tokenizer.json"))

        if self.X_train is not None:
            np.savez_compressed(
                os.path.join(dest_folder, "train_data_nn.npz"),
                X=self.X_train,
                y=self.y_train,
            )

        if self.X_test is not None:
            np.savez_compressed(
                os.path.join(dest_folder, "test_data_nn.npz"),
                X=self.X_test,
                y=self.y_test,
            )

        if self.X_val is not None:
            np.savez_compressed(
                os.path.join(dest_folder, "val_data_nn.npz"), X=self.X_val, y=self.y_val
            )

        logging.info(
            "Data Export Completed - Time elapsed: " + get_elapsed_time(start_time)
        )

    def from_numpy(
        self,
        train_data_file: str,
        test_data_file: str = None,
        val_data_file: str = None,
        ds_type="TensorDataset",
    ):

        logging.info("Starting Data Preparation...")
        start_time = time.time()

        self.tokenizer = text.tokenizer_from_json()

        train_npz = np.load(train_data_file, allow_pickle=True)
        self.X_train = train_npz["X"].item()
        self.y_train = train_npz["y"]

        self.num_classes = len(np.unique(self.y_train))
        self.vocab_size = np.shape(self.X_train)[1]

        train_ds = CSRDataset(self.X_train, self.y_train)

        if test_data_file is not None:
            test_npz = np.load(test_data_file, allow_pickle=True)
            self.X_test = test_npz["X"].item()
            self.y_test = test_npz["y"]

            test_ds = CSRDataset(self.X_test, self.y_test)

        if val_data_file is not None:
            val_npz = np.load(val_data_file, allow_pickle=True)
            self.X_val = val_npz["X"].item()
            self.y_val = val_npz["y"]

            val_ds = CSRDataset(self.X_val, self.y_val)

        logging.info(
            "Data Import Completed - Time elapsed: " + get_elapsed_time(start_time)
        )

        if val_data_file is not None:
            if test_data_file is not None:
                return train_ds, val_ds, test_ds
            else:
                return train_ds, val_ds
        else:
            return train_ds


class DSGenerator(object):
    def __init__(self, seed: int = 123):

        self.seed = seed
        self.train_file = None
        self.test_file = None
        self.val_file = None

        self.text_col_idx = None
        self.label_col_idx = None

        self.vectorizer = None
        self.vocab = None
        self.vocab_size = None
        self.num_classes = None

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_val = None
        self.y_val = None

        self.params = {}

        logging.getLogger(__name__)

    def from_csv(
        self,
        train_file: str,
        test_file: str = None,
        val_file: str = None,
        val_size: float = 0.0,
        text_col_idx=0,
        label_col_idx=1,
        sep: str = ",",
        header=0,
        encoding: str = "utf8",
        preprocess_func=None,
        preprocess_ncore=2,
        ngram_range=(1, 3),
        min_df=1,
        max_df=1.0,
        stop_words="english",
        max_features=20000,
        ds_max_seq=1000,
        ds_type="TensorDataset",
    ):

        logging.info("Starting Data Preparation ...")
        logging.info("  Training Data ...")
        start_time = time.time()

        np.random.seed(self.seed)
        self.vectorizer = DataVectorizer(
            preprocess_func,
            preprocess_ncore,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            stop_words=stop_words,
        )

        self.train_file = train_file
        self.test_file = test_file
        self.val_file = val_file

        self.text_col_idx = text_col_idx
        self.label_col_idx = label_col_idx

        df = pd.read_csv(self.train_file, sep=sep, encoding=encoding, header=header)
        X = df[df.columns[self.text_col_idx]].tolist()
        y = df[df.columns[self.label_col_idx]].to_numpy(dtype=int)
        del df

        if val_size > 0.0 and self.val_file is None:
            # create valid partition from train partition, keeping class distribution
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, stratify=y, test_size=val_size, random_state=self.seed
            )
            X = X_train
            y = y_train
            del X_train, y_train
            gc.collect()

        # Input features features
        X = self.vectorizer.fit_transform(X)
        self.X_train = sequence.pad_sequences(X, maxlen=ds_max_seq, padding="post")
        self.y_train = y
        del X, y
        gc.collect()

        self.vocab_size = self.vectorizer.vocab_size
        self.vocab = self.vectorizer.w2index
        self.num_classes = len(list(set(self.y_train)))

        if ds_type == "TensorDataset":
            train_ds = TensorDataset(
                torch.from_numpy(self.X_train).long(),
                torch.from_numpy(self.y_train).long(),
            )
        else:
            train_ds = NNDataset(self.X_train, self.y_train, max_seq=ds_max_seq)

        if self.test_file is not None:
            logging.info("  Test Data ...")
            df = pd.read_csv(self.test_file, sep=sep, encoding=encoding, header=header)
            X = df[df.columns[self.text_col_idx]].tolist()
            y = df[df.columns[self.label_col_idx]].to_numpy(dtype=int)
            del df

            X = self.vectorizer.transform(X)
            self.X_test = sequence.pad_sequences(X, maxlen=ds_max_seq)
            self.y_test = y
            del X, y
            gc.collect()

            if ds_type == "TensorDataset":
                test_ds = TensorDataset(
                    torch.from_numpy(self.X_test).long(),
                    torch.from_numpy(self.y_test).long(),
                )
            else:
                test_ds = NNDataset(self.X_test, self.y_test, max_seq=ds_max_seq)

        if (val_size > 0.0 and self.val_file is None) or self.val_file is not None:
            logging.info("  Validation Data ...")
            if self.val_file is not None:
                df = pd.read_csv(self.val_file, sep=sep, encoding=encoding)
                X_val = df[df.columns[self.text_col_idx]].tolist()
                y_val = df[df.columns[self.label_col_idx]].to_numpy(dtype=int)
                del df
            X_val = self.vectorizer.transform(X_val)
            self.X_val = sequence.pad_sequences(X_val, maxlen=ds_max_seq)
            self.y_val = y_val
            del X_val, y_val
            gc.collect()

            if ds_type == "TensorDataset":
                val_ds = TensorDataset(
                    torch.from_numpy(self.X_val).long(),
                    torch.from_numpy(self.y_val).long(),
                )
            else:
                val_ds = NNDataset(self.X_val, self.y_val, max_seq=ds_max_seq)

        logging.info(
            "Data Preparation Completed - Time elapsed: " + get_elapsed_time(start_time)
        )

        self.params = {
            "seed": self.seed,
            "train_file": self.train_file,
            "test_file": self.test_file,
            "val_file": self.val_file,
            "vocabulary_size":  self.vocab_size,
            "preprocess_ncore": preprocess_ncore,
            "stop_words": stop_words,
            "max_features": max_features,
            "ngram_range": ngram_range,
            "min_df": min_df,
            "max_df": max_df,
            "ds_max_seq": ds_max_seq,
            "num_classes": self.num_classes
        }
        if preprocess_func is not None:
            self.params.update({"preprocess_func": preprocess_func.__name__})
        else:
            self.params.update({"preprocess_func": None})

        if self.val_file is not None:
            if self.test_file is not None:
                return train_ds, test_ds, val_ds
            else:
                return train_ds, val_ds
        else:
            return train_ds
