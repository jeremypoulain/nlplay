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
