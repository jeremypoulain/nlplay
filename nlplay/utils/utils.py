"""Various utils"""
import logging
import math
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import yaml
from sklearn.decomposition import PCA
from tqdm import tqdm


def _loguniform(min_val, max_val, n=5):
    out = []
    _min = math.log2(float(min_val))
    _max = math.log2(float(max_val))
    for i in range(n):
        out.append(2 ** random.uniform(_min, _max))
    return out


def get_topk_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, k: int = 3, include_details=False
):
    """
    Vectorized top K accuracy.
    :param y_true: true labels of shape (n_samples,).
    :param y_pred: predicted labels ranked from the most to the least likely, shape (n_samples, n_labels).
        Scores or probabilities must be ranked first, e.g. with np.argsort(-scores, axis=1).
    :param k: number of top ranked labels considered.
    :param include_details: also return the total, correct and wrong counts.
    :returns: the top K accuracy, or (accuracy, total, ok, ko) if include_details is True.
    """
    if len(y_true) == 0:
        raise ValueError("y_true is empty")

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


def _write_stream(response: requests.Response, destination, mode: str, total, initial: int = 0, desc: str = ""):
    with open(destination, mode) as file, tqdm(
        total=total, initial=initial, unit="B", unit_scale=True, desc=desc
    ) as pbar:
        for chunk in response.iter_content(chunk_size=1024 * 64):
            # filter out keep-alive new chunks
            if chunk:
                file.write(chunk)
                pbar.update(len(chunk))


def download_file_from_google_drive(file_id: str, destination: str, file_size: int | None = None, timeout: float = 60):
    """
    Download a publicly shared Google Drive file, including large files behind the virus scan warning.
    :param file_id: Google Drive file id.
    :param destination: output file path.
    :param file_size: expected size in bytes, only used for the progress bar.
    :param timeout: connection and read timeout in seconds.
    """
    # confirm=t skips the virus scan warning page served for large files
    url = "https://drive.usercontent.google.com/download"
    params = {"id": file_id, "export": "download", "confirm": "t"}
    with requests.get(url, params=params, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        # An HTML page means an error or warning page instead of the file content
        if response.headers.get("Content-Type", "").startswith("text/html"):
            raise RuntimeError(
                f"Google Drive returned an HTML page instead of file {file_id}, "
                "check that it is shared publicly and that its download quota is not exceeded"
            )
        _write_stream(response, destination, "wb", file_size, desc="Downloading Dataset...")


def download_from_url(url: str, dst: Path, timeout: float = 60) -> int:
    """
    Download a file, resuming a partial download when the server supports range requests.
    :param url: file url.
    :param dst: output file path.
    :param timeout: connection and read timeout in seconds.
    :returns: the size of the downloaded file in bytes.
    """
    dst = Path(dst)
    first_byte = dst.stat().st_size if dst.exists() else 0
    headers = {"Range": f"bytes={first_byte}-"} if first_byte > 0 else {}

    with requests.get(url, headers=headers, stream=True, timeout=timeout) as response:
        if response.status_code == 416:
            # Nothing left to download, unless the local file does not match the remote size
            total = response.headers.get("Content-Range", "").rpartition("/")[2]
            if not total.isdigit() or int(total) == first_byte:
                return first_byte
            dst.unlink()
            return download_from_url(url, dst, timeout)
        response.raise_for_status()

        if response.status_code == 206:
            # Partial content → append the missing bytes, Content-Range is "bytes start-end/total"
            total = response.headers.get("Content-Range", "").rpartition("/")[2]
            _write_stream(response, dst, "ab", int(total) if total.isdigit() else None, first_byte,
                          desc="Downloading Dataset...")
        else:
            # The server ignored the Range header and sends the whole file → restart from scratch
            length = response.headers.get("Content-Length")
            _write_stream(response, dst, "wb", int(length) if length else None, desc="Downloading Dataset...")

    return dst.stat().st_size


def read_config(config_file_path: str):
    with open(config_file_path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def get_elapsed_time(start_time: float):
    """
    Compute and format the elapsed time since start_time.
    :param start_time: start time as returned by time.time().
    :returns: elapsed time formatted as "Xm Ys", or "Xh Ym Zs" above one hour.
    """
    s = time.time() - start_time
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return "%dh %dm %ds" % (h, m, s)
    return "%dm %ds" % (m, s)


def human_readable_size(size: int, decimal_places=2):
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if size < 1024.0:
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f}{unit}"


def df_optimize(df: pd.DataFrame, catg_conv_threshold: float = 0.4, downcast_float: bool = True):
    """
    Reduce the memory usage of a dataframe, in place.
    Adapted from https://www.kaggle.com/nilanml/imdb-review-deep-model-94-89-accuracy
    Integers → smallest integer type holding all their values.
    Floats → float32 if downcast_float, never float16 which only keeps about 3 significant digits.
    Strings → category when the ratio of unique values is below catg_conv_threshold.
    :param df: input dataframe, modified in place.
    :param catg_conv_threshold: max ratio of unique values to convert a string column to category.
    :param downcast_float: convert float64 columns to float32, which keeps about 7 significant digits.
    :returns: the optimized dataframe.
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    df_size = df.shape[0]
    for col in df.columns:
        col_type = df[col].dtype
        if pd.api.types.is_bool_dtype(col_type):
            continue
        if pd.api.types.is_integer_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="integer" if df[col].min() < 0 else "unsigned")
        elif pd.api.types.is_float_dtype(col_type):
            if downcast_float:
                df[col] = pd.to_numeric(df[col], downcast="float")
        elif col_type == object or isinstance(col_type, pd.StringDtype):
            # object for pandas < 3, str (StringDtype) since pandas 3
            if df_size > 0 and df[col].nunique() / df_size <= catg_conv_threshold:
                df[col] = df[col].astype("category")

    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    logging.info("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    if start_mem > 0:
        logging.info("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


def postprocess_pretrained_vecs(
    in_vec_file: str = "", out_vec_filepath: str = "", N: int = 2, strip_pos_suffix: bool = False
):
    """
    Title   : All-but-the-Top: Simple and Effective Postprocessing for Word Representations - 2017
    Author  : Jiaqi Mu, Suma Bhat, Pramod Viswanath
    Papers  : https://arxiv.org/pdf/1702.01417
    Source  : https://blogs.nlmatics.com/nlp/sentence-embeddings/2020/08/07/Smooth-Inverse-Frequency-Frequency-(SIF)-Embeddings-in-Golang.html
    Note    : Removes the common mean vector and the top N principal components from every word vector.
              Reads and writes the text format (word2vec with a "count dim" header, or GloVe without).
    :param in_vec_file: input text vectors file.
    :param out_vec_filepath: output text vectors file, with a header if the input had one.
    :param N: number of top principal components to remove.
    :param strip_pos_suffix: remove a "_POS" suffix from the words, e.g. "run_VERB" → "run".
    """
    words, vectors, header, dim = [], [], False, None
    with open(in_vec_file, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f):
            parts = line.rstrip().split(" ")
            if not parts[0]:
                continue
            # word2vec header "count dim"
            if line_no == 0 and len(parts) == 2 and all(p.isdigit() for p in parts):
                header = True
                continue
            if dim is None:
                dim = len(parts) - 1
            if len(parts) < dim + 1:
                raise ValueError(f"Line {line_no + 1} has {len(parts) - 1} values, expected {dim}")
            # The last dim fields are the vector, some GloVe tokens contain spaces
            word = " ".join(parts[:-dim])
            if strip_pos_suffix:
                word = word.rsplit("_", 1)[0]
            words.append(word)
            vectors.append(np.asarray(parts[-dim:], dtype=np.float32))

    if not vectors:
        raise ValueError(f"No vectors found in {in_vec_file}")
    if not 0 <= N < dim:
        raise ValueError(f"N must be in [0, {dim}), got {N}")

    # Subtract the average vector, then remove the projections on the top N principal components
    embs = np.stack(vectors)
    embs -= embs.mean(axis=0)
    if N > 0:
        components = PCA(n_components=N).fit(embs).components_
        embs -= (embs @ components.T) @ components

    # write back new word vector file
    with open(out_vec_filepath, "w", encoding="utf-8") as file:
        if header:
            file.write(f"{len(words)} {dim}\n")
        for word, vec in zip(words, embs):
            file.write(word + " " + " ".join(f"{x:.7g}" for x in vec) + "\n")
