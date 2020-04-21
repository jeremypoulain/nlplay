import logging
import re
import string
import time
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from nlplay.utils.parlib import parallelApply
from nlplay.utils.utils import get_elapsed_time


class DataVectorizer(object):
    def __init__(self, preprocessing_function=None, preprocess_ncore=4,
                 ngram_range=(1, 2), min_df=5, max_df=0.95, stop_words='english', max_features=200000,
                 use_unk_tok=False, encode_label=False):
        """

        :param preprocessing_function:
        :param preprocess_ncore:
        :param ngram_range:
        :param min_df:
        :param max_df:
        :param stop_words:
        :param max_features:
        :param use_unk_tok:
        :param encode_label:
        """
        self.vectorizer = None
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.stop_words = stop_words

        self.labels = None
        self.label_encoder = None
        self.encode_label = encode_label

        self.preprocess_func = preprocessing_function
        self.preprocess_ncore = preprocess_ncore

        self.vocab_size = None
        self.use_unk_tok = use_unk_tok

        self.padding_token = '<pad>'
        self.unknown_token = '<unk>'
        self.eos_token = '</s>'
        self.sos_token = '<s>'
        self.reserved_tokens = [self.padding_token, self.unknown_token, self.eos_token, self.sos_token]
        self.index2w = self.reserved_tokens
        self.w2index = {token: index for index, token in enumerate(self.reserved_tokens)}
        self.unknown_index = self.w2index['<unk>']
        self.padding_index = self.w2index['<pad>']

        logging.getLogger(__name__)

    def load(self):
        pass

    def save(self):
        pass

    def fit_transform(self, X, y=None):
        logging.info("DataVectorizer.fit_transform running...")
        start_time = time.time()

        # Apply pre-processing function
        if self.preprocess_func:
            _X = parallelApply(X, self.preprocess_func, self.preprocess_ncore)
        else:
            _X = X
        if y is not None:
            if self.encode_label:
                self.label_encoder = LabelEncoder()
                _y = self.label_encoder.fit_transform(y)
                self.labels = self.label_encoder.classes_
            else:
                _y = np.asarray(y)

        if self.preprocess_func is None:
            re_tok = re.compile('([%s“”¨«»®´·º½¾¿¡§£₤‘’])' % string.punctuation)
            tokenizer = lambda x: re_tok.sub(r' \1 ', x).split()
            self.vectorizer = CountVectorizer(tokenizer=tokenizer,
                                              ngram_range=self.ngram_range, min_df=self.min_df, max_df=self.max_df,
                                              max_features=self.max_features, stop_words=self.stop_words,
                                              lowercase=True)
        else:
            self.vectorizer = CountVectorizer(tokenizer=None,
                                              ngram_range=self.ngram_range, min_df=self.min_df, max_df=self.max_df,
                                              max_features=self.max_features, stop_words=self.stop_words)
        self.vectorizer.fit_transform(_X)
        kept_tokens = set(self.vectorizer.vocabulary_.keys())

        for token in kept_tokens:
            self.index2w.append(token)
            self.w2index[token] = len(self.index2w) - 1
        self.vocab_size = len(self.index2w)
        del kept_tokens

        # transform sentences into list of word indexes
        _X = self.texts_to_sequences(_X)

        logging.info("   DataVectorizer.fit_transform completed - Time elapsed: " + get_elapsed_time(start_time))

        if y is not None:
            return _X, _y
        else:
            return _X

    def transform(self, X, y=None):

        logging.info("   DataVectorizer.transform running...")
        start_time = time.time()

        # Apply pre-processing function
        if self.preprocess_func:
            _X = parallelApply(X, self.preprocess_func, self.preprocess_ncore)
        else:
            _X = X
        if y is not None and self.encode_label:
            _y = self.label_encoder.transform(y)
        else:
            _y = np.asarray(y)

        # transform sentences into list of word indexes
        self.vectorizer.transform(X)
        _X = self.texts_to_sequences(_X)

        logging.info("   DataVectorizer.transform completed - Time elapsed: " + get_elapsed_time(start_time))

        if y is not None:
            return _X, _y
        else:
            return _X

    def texts_to_sequences(self, texts):
        """Transforms each text in texts to a sequence of integers.
        # Arguments
            texts: A list of texts (strings).

        # Returns
            A list of sequences.
        """
        return list(self.texts_to_sequences_generator(texts))

    def texts_to_sequences_generator(self, texts):
        for text in texts:
            if self.use_unk_tok:
                yield [self.w2index.get(word, self.unknown_index) for word in text.split()]
            else:
                yield [self.w2index[word] for word in text.split() if word in self.w2index]

    def sequences_to_texts(self, sequences):
        """Transforms each sequence into a list of text.

        # Arguments
            sequences: A list of sequences (list of integers).

        # Returns
            A list of texts (strings)
        """
        return list(self.sequences_to_texts_generator(sequences))

    def sequences_to_texts_generator(self, sequences):
        for seq in sequences:
            if self.use_unk_tok:
                yield [self.index2w.get(idx, self.unknown_index) for idx in seq]
            else:
                yield [self.index2w[idx] for idx in seq]