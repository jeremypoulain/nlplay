import logging
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from nlplay.utils.utils import get_elapsed_time
from datetime import datetime


def get_pretrained_vecs(
    input_vec_file: str, target_vocab: dict, dim: int = 300, output_file=None
):

    logging.getLogger(__name__)
    start_time = time.time()
    found_words = 0
    missing_words = 0

    # import word vector text file into a pandas dataframe (quicker)
    df_wvecs = pd.read_csv(input_vec_file, sep=" ", quoting=3, header=None, index_col=0)

    # create word index dict from pretrained vectors
    w_index = df_wvecs.index.tolist()
    w_index = dict(zip(w_index, range(len(w_index))))
    np_w_vecs = df_wvecs.to_numpy()
    del df_wvecs

    # initialize embedding matrix weights
    emb_mean, emb_std = np_w_vecs.mean(), np_w_vecs.std()
    embedding_matrix = np.random.normal(emb_mean, emb_std, (len(target_vocab), dim))
    embedding_matrix[0] = np.zeros(dim)

    # recopy pretrained vect weights into embedding_matrix
    # TODO: vectorize the following code
    for k, v in tqdm(
        target_vocab.items(),
        desc="{} Processing pretrained vectors...".format(
            datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        ),
        total=len(target_vocab),
    ):
        if k in w_index.keys():
            found_words += 1
            embedding_matrix[v] = np_w_vecs[w_index[k]]
        else:
            missing_words += 1

    if output_file is not None:
        np.save(output_file, embedding_matrix)

    logging.info(
        "Matching words: {} - input vocab: {} - coverage: {}".format(
            found_words, len(target_vocab), found_words / len(target_vocab)
        )
    )
    logging.info(
        "Pretrained Vectors Preparation - Completed - Time elapsed: "
        + get_elapsed_time(start_time)
    )

    return embedding_matrix
