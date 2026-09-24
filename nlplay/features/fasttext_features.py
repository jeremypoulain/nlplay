"""
Title    : Bag of Tricks for Efficient Text Classification - 2017
Authors  : Armand Joulin, Edouard Grave, Piotr Bojanowski, Tomas Mikolov
Papers   : https://aclanthology.org/E17-2068.pdf
Source   : https://github.com/facebookresearch/fastText/blob/main/src/dictionary.cc

Note     : Feature extraction of the fastText C++ supervised mode, id for id: each text becomes one
           flat bag of feature ids made of
             - the vocabulary word ids (0 .. nwords - 1),
             - the hashed character n-grams of each word (minn .. maxn, words wrapped in < >),
               out of vocabulary words only contribute their character n-grams,
             - the hashed word n-grams (2 .. word_ngrams) appended after the words,
           hashed into bucket ids nwords .. nwords + bucket - 1, with the same 32 bit FNV-1a hash
           over signed bytes and the same 116049371 rolling hash as fastText, so that the ids of a
           native fastText model built with the same words can be reproduced exactly.
"""
import re
from collections import Counter
from collections.abc import Iterable

import numpy as np

EOS = "</s>"
BOW = "<"
EOW = ">"
LABEL_PREFIX = "__label__"
_UINT32 = 0xFFFFFFFF
_UINT64 = 0xFFFFFFFFFFFFFFFF
# Separators of fastText readWord
_SPLIT = re.compile(r"[ \n\r\t\v\f\0]+")


def fasttext_hash(data: bytes) -> int:
    """
    32 bit FNV-1a hash of fastText, which xors the bytes as signed chars.
    :param data: utf-8 encoded string.
    :returns: the unsigned 32 bit hash.
    """
    h = 2166136261
    for byte in data:
        h = ((h ^ ((byte - 256 if byte > 127 else byte) & _UINT32)) * 16777619) & _UINT32
    return h


def read_word_vectors(path: str) -> tuple[list[str], np.ndarray]:
    """
    Read text word vectors, word2vec / fastText .vec format with a "count dim" header, or GloVe without.
    :param path: vectors file.
    :returns: (words, float32 matrix of shape (n_words, dim)).
    """
    words, vectors, dim = [], [], None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f):
            parts = line.rstrip().split(" ")
            if not parts[0]:
                continue
            if line_no == 0 and len(parts) == 2 and all(p.isdigit() for p in parts):
                continue
            if dim is None:
                dim = len(parts) - 1
            if len(parts) < dim + 1:
                raise ValueError(f"Line {line_no + 1} has {len(parts) - 1} values, expected {dim}")
            # The last dim fields are the vector, some GloVe tokens contain spaces
            words.append(" ".join(parts[:-dim]))
            vectors.append(np.asarray(parts[-dim:], dtype=np.float32))
    if not vectors:
        raise ValueError(f"No vectors found in {path}")
    return words, np.stack(vectors)


def _sign_extend(h: int) -> int:
    # fastText stores the uint32 word hashes in an int32 vector, then reads them back as uint64
    return (h - (1 << 32) if h >= 1 << 31 else h) & _UINT64


class FastTextFeaturizer:
    def __init__(
        self,
        word_ngrams: int = 2,
        minn: int = 0,
        maxn: int = 0,
        bucket: int = 2_000_000,
        min_count: int = 1,
        add_eos: bool = True,
    ):
        """
        :param word_ngrams: max length of the hashed word n-grams, 1 → words only (fastText -wordNgrams).
        :param minn: min length of the character n-grams (fastText -minn).
        :param maxn: max length of the character n-grams, 0 → no character n-grams (fastText -maxn).
        :param bucket: number of hash buckets shared by the word and character n-grams (fastText -bucket).
        :param min_count: minimal count of a vocabulary word (fastText -minCount).
        :param add_eos: end each text with the </s> token, as fastText does for every line.
        """
        if word_ngrams < 1 or bucket < 1 or min_count < 1:
            raise ValueError("word_ngrams, bucket and min_count must be >= 1")
        if maxn > 0 and not 1 <= minn <= maxn:
            raise ValueError("minn must be in [1, maxn] when maxn > 0")
        self.word_ngrams = word_ngrams
        self.minn = minn
        self.maxn = maxn
        self.bucket = bucket
        self.min_count = min_count
        self.add_eos = add_eos
        self.words_: list[str] = []
        self.word2id_: dict[str, int] = {}
        self._cache: dict[str, tuple[int, list[int]]] = {}

    @property
    def nwords(self) -> int:
        return len(self.words_)

    @property
    def num_features(self) -> int:
        """Size of the input embedding table: vocabulary words + hash buckets."""
        return self.nwords + self.bucket

    def tokenize(self, text: str) -> list[str]:
        """
        :param text: raw text, split on the fastText separators, label tokens removed.
        :returns: the word tokens, ending with </s> if add_eos.
        """
        tokens = [t for t in _SPLIT.split(text) if t and not t.startswith(LABEL_PREFIX)]
        if self.add_eos:
            tokens.append(EOS)
        return tokens

    def fit(self, texts: Iterable[str], pretrained_words: Iterable[str] | None = None) -> "FastTextFeaturizer":
        """
        Build the vocabulary, words sorted by decreasing count (ties by first occurrence, the C++ order
        of ties is unspecified), words below min_count dropped.
        :param texts: training texts.
        :param pretrained_words: words of pretrained vectors (read_word_vectors), added to the vocabulary
            with one extra count and min_count lowered to 1, as fastText -pretrainedVectors does.
        :returns: self.
        """
        counts = Counter(t for text in texts for t in self.tokenize(text))
        min_count = self.min_count
        if pretrained_words is not None:
            counts.update(w for w in pretrained_words if not w.startswith(LABEL_PREFIX))
            min_count = 1
        words = [w for w, c in sorted(counts.items(), key=lambda wc: -wc[1]) if c >= min_count]
        return self._set_words(words)

    @classmethod
    def from_words(cls, words: Iterable[str], **kwargs) -> "FastTextFeaturizer":
        """
        :param words: vocabulary in id order, e.g. native_model.get_words(include_freq=False).
        :param kwargs: FastTextFeaturizer parameters, matching the native model ones.
        :returns: a featurizer producing the ids of that vocabulary.
        """
        return cls(**kwargs)._set_words(list(words))

    def _set_words(self, words: list[str]) -> "FastTextFeaturizer":
        self.words_ = words
        self.word2id_ = {w: i for i, w in enumerate(words)}
        self._cache = {}
        return self

    def _char_ngrams(self, word: str) -> list[int]:
        # computeSubwords over the utf-8 bytes of <word>, n-grams counted in characters
        data = (BOW + word + EOW).encode("utf-8")
        size, ids = len(data), []
        for i in range(size):
            if (data[i] & 0xC0) == 0x80:
                continue
            j, n = i, 1
            while j < size and n <= self.maxn:
                j += 1
                while j < size and (data[j] & 0xC0) == 0x80:
                    j += 1
                if n >= self.minn and not (n == 1 and (i == 0 or j == size)):
                    ids.append(self.nwords + fasttext_hash(data[i:j]) % self.bucket)
                n += 1
        return ids

    def _word_features(self, token: str) -> tuple[int, list[int]]:
        # (word hash, word id and / or character n-gram ids), cached per token
        cached = self._cache.get(token)
        if cached is None:
            wid = self.word2id_.get(token)
            if wid is None:
                ids = self._char_ngrams(token) if self.maxn > 0 and token != EOS else []
            else:
                ids = [wid] + (self._char_ngrams(token) if self.maxn > 0 and token != EOS else [])
            cached = (_sign_extend(fasttext_hash(token.encode("utf-8"))), ids)
            self._cache[token] = cached
        return cached

    def features(self, text: str) -> list[int]:
        """
        :param text: raw text.
        :returns: the feature ids of the text, in the fastText order.
        """
        ids, hashes = [], []
        for token in self.tokenize(text):
            h, word_ids = self._word_features(token)
            ids.extend(word_ids)
            hashes.append(h)
        # addWordNgrams: rolling hash over the next word_ngrams - 1 words
        for i in range(len(hashes)):
            h = hashes[i]
            for j in range(i + 1, min(len(hashes), i + self.word_ngrams)):
                h = (h * 116049371 + hashes[j]) & _UINT64
                ids.append(self.nwords + h % self.bucket)
        return ids

    def transform(self, texts: Iterable[str]) -> list[np.ndarray]:
        """
        :param texts: raw texts.
        :returns: one int64 array of feature ids per text.
        """
        if not self.words_:
            raise ValueError("The featurizer must be fitted first")
        return [np.asarray(self.features(text), dtype=np.int64) for text in texts]

    def fit_transform(self, texts: Iterable[str]) -> list[np.ndarray]:
        texts = list(texts)
        return self.fit(texts).transform(texts)
