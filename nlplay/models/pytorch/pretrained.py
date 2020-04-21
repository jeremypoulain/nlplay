import logging
import time
import numpy as np
from tqdm import tqdm
from nlplay.utils.utils import get_elapsed_time
from datetime import datetime


def get_pretrained_vecs(input_vec_file: str, target_vocab: dict, dim: int = 300,
                        unk_range: float = 0.25, output_file=None):

    logging.getLogger(__name__)
    start_time = time.time()
    found_words = 0
    missing_words = 0
    embeddings = np.random.uniform(low=-unk_range, high=unk_range, size=(len(target_vocab), dim))
    embeddings[0] = np.zeros(dim)

    with open(input_vec_file, "r", encoding="utf8") as f:
        for index, line in enumerate(tqdm(f, desc='{} Processing pretrained vectors...'.format(
                    datetime.today().strftime('%Y-%m-%d %H:%M:%S')))):
            values = line.split()
            word = values[0]
            if word in target_vocab:
                w_vec = np.asarray(values[1:], dtype='float32')
                embeddings[target_vocab[word]:] = w_vec
                found_words += 1

    logging.info('Found words: {} - input vocab: {} - coverage: {}'.format(found_words, len(target_vocab),
                                                                           found_words/len(target_vocab)))

    if output_file is not None:
        np.save(output_file, embeddings)

    logging.info("Pretrained Vectors Preparation - Completed - Time elapsed: " + get_elapsed_time(start_time))

    return embeddings