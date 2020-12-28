"""Various utils"""
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen
import numpy as np
import pandas as pd
import requests
import yaml
from sklearn.decomposition import PCA
from tqdm import tqdm


def _loguniform(min_val, max_val, len=5):
    out = []
    _min = math.log2(float(min_val))
    _max = math.log2(float(max_val))
    for i in range(len):
        out.append(2 ** random.uniform(_min, _max))
    return out


def get_topk_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, k: int = 3, include_details=False
):
    """
    vectorized function to compute top K accuracy
    """
    # Only keep K first columns
    inscope_ypred = y_pred[:, 0:k]

    # Check for any match along each row
    out = ((inscope_ypred == y_true[:, None]).any(1)).astype(int)

    # Compute final topk accuracy score
    total = len(y_true)
    ok = np.sum(out == 1)
    ko = total - ok
    acc_score = ok / total

    if include_details:
        return acc_score, total, ok, ko
    else:
        return acc_score


def download_file_from_google_drive(id: str, destination: str, file_size: int):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value

        return None

    def save_response_content(response, destination, file_size):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as file:
            size = file_size
            pieces = int(size / CHUNK_SIZE)
            with tqdm(
                total=pieces,
                desc="{} Downloading Dataset...".format(
                    datetime.today().strftime("%Y-%m-%d %H:%M:%S")
                ),
                unit="B",
            ) as pbar:
                for chunk in response.iter_content(CHUNK_SIZE):
                    pbar.update(1)
                    if chunk:  # filter out keep-alive new chunks
                        file.write(chunk)

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination, file_size)


def download_from_url(url: str, dst: Path):
    file_size = int(urlopen(url).info().get("Content-Length", -1))
    if os.path.exists(dst):
        first_byte = os.path.getsize(dst)
    else:
        first_byte = 0
    if first_byte >= file_size:
        return file_size
    header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
    pbar = tqdm(
        total=file_size,
        initial=first_byte,
        unit="B",
        unit_scale=True,
        desc="Downloading Dataset...",
    )
    req = requests.get(url, headers=header, stream=True)
    with (open(dst, "ab")) as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(1024)
    pbar.close()

    return file_size


def read_config(config_file_path: str = None):
    with open(config_file_path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def get_elapsed_time(start_time: float):
    """
    Compute & format elapsed time between a start & stop time
    """
    now = time.time()
    s = now - start_time
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def human_readable_size(size: int, decimal_places=2):
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if size < 1024.0:
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f}{unit}"


def df_optimize(df: pd.DataFrame, catg_conv_threshold: float = 0.4):
    """
    Optimize pandas dataframe memory usage for numerical and string features
    Adapted from https://www.kaggle.com/nilanml/imdb-review-deep-model-94-89-accuracy
    df : INput dataframe
    catg_conv_threshold : threshold use to trigger the conversion of string object to categorical ones
    # TODO: Null values check for numerics!
            Column sparsity?
    """

    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    strings = ["object"]
    df_size = df.shape[0]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                # TODO: Null values check !
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

        if col_type in strings:
            if (df[col].nunique() / df_size) <= catg_conv_threshold:
                df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


def postprocess_pretrained_vecs(
    in_vec_file: str = "", out_vec_filepath: str = "", N: int = 2
):
    """
    Title   : All-but-the-Top: Simple and Effective Postprocessing for Word Representations - 2017
    Author  : Jiaqi Mu, Suma Bhat, Pramod Viswanath
    Papers  : https://arxiv.org/pdf/1702.01417
    Source  : https://blogs.nlmatics.com/nlp/sentence-embeddings/2020/08/07/Smooth-Inverse-Frequency-Frequency-(SIF)-Embeddings-in-Golang.html
    """
    embs = []

    # map indexes of word vectors in matrix to their corresponding words
    idx_to_word = dict()
    dimension = 0

    # append each vector to a 2-D matrix and calculate average vector
    with open(in_vec_file, "rb") as f:
        first_line = []
        for line in f:
            first_line = line.rstrip().split()
            dimension = len(first_line) - 1
            if dimension < 100:
                continue
            print("dimension: ", dimension)

            break
        avg_vec = [0] * dimension
        vocab_size = 0
        word = str(first_line[0].decode("utf-8"))
        word = word.split("_")[0]

        idx_to_word[vocab_size] = word
        vec = [float(x) for x in first_line[1:]]
        avg_vec = [vec[i] + avg_vec[i] for i in range(len(vec))]
        vocab_size += 1
        embs.append(vec)
        for line in f:
            line = line.rstrip().split()
            word = str(line[0].decode("utf-8"))
            word = word.split("_")[0]
            idx_to_word[vocab_size] = word
            vec = [float(x) for x in line[1:]]
            avg_vec = [vec[i] + avg_vec[i] for i in range(len(vec))]
            vocab_size += 1
            embs.append(vec)
        avg_vec = [x / vocab_size for x in avg_vec]
    # convert to numpy array
    embs = np.array(embs)

    # subtract average vector from each vector
    for i in range(len(embs)):
        new_vec = [embs[i][j] - avg_vec[j] for j in range(len(avg_vec))]
        embs[i] = np.array(new_vec)

    # principal component analysis using sklearn
    pca = PCA()
    pca.fit(embs)

    # remove the top N components from each vector
    for i in range(len(embs)):
        preprocess_sum = [0] * dimension
        for j in range(N):
            princip = np.array(pca.components_[j])
            preprocess = princip.dot(embs[i])
            preprocess_vec = [princip[k] * preprocess for k in range(len(princip))]
            preprocess_sum = [
                preprocess_sum[k] + preprocess_vec[k]
                for k in range(len(preprocess_sum))
            ]
        embs[i] = np.array(
            [embs[i][j] - preprocess_sum[j] for j in range(len(preprocess_sum))]
        )

    # write back new word vector file
    file = open(out_vec_filepath, "w+", encoding="utf-8")
    idx = 0
    for vec in embs:
        file.write(idx_to_word[idx])
        file.write(" ")
        for num in vec:
            file.write(str(num))
            file.write(" ")
        file.write("\n")
        idx += 1
    file.close()
