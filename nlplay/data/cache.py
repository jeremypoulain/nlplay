import os
import zipfile
import tarfile
import logging
import pandas as pd
from enum import Enum
from pathlib import Path
from nlplay.utils import utils
from nlplay.utils.utils import download_file_from_google_drive, download_from_url


class WV(Enum):
    FASTTEXT_XXWIKI_300 = (
        "FASTTEXT_XXWIKI_300"
    )  # Where XX is the language code ie FR-French
    FASTTEXT_ENWIKI_300 = "FASTTEXT_ENWIKI_300"
    FASTTEXT_ENWIKI_SUB_300 = "FASTTEXT_ENWIKI_SUB_300"
    FASTTEXT_ENCC_300 = "FASTTEXT_ENCC_300"
    FASTTEXT_ENCC_SUB_300 = "FASTTEXT_ENCC_SUB_300"
    GLOVE_EN6B_50 = "GLOVE_EN6B_50"
    GLOVE_EN6B_100 = "GLOVE_EN6B_100"
    GLOVE_EN6B_200 = "GLOVE_EN6B_200"
    GLOVE_EN6B_300 = "GLOVE_EN6B_300"
    GLOVE_EN840B_300 = "GLOVE_EN840B_300"


class DS(Enum):
    def __str__(self):
        return str(self.value)

    TWENTY_NEWS = "20_NEWS"
    AG_NEWS = "AG_NEWS"
    AMAZON_FULL = "AMAZON_FULL"
    AMAZON_POLARITY = "AMAZON_POLARITY"
    DBPEDIA = "DBPEDIA"
    IMDB = "IMDB"
    MR = "MR"
    SOGOU = "SOGOU"
    SST2 = "SST2"
    TREC_6 = "TREC_6"
    TREC_50 = "TREC_50"
    YAHOO = "YAHOO"
    YELP_FULL = "YELP_FULL"
    YELP_POLARITY = "YELP_POLARITY"


class WordVectorsManager(object):
    """
    Class to automatically download & send back a Glove/Fasttext/... pretrained vectors text file
    """
    def __init__(self, model_name=WV.FASTTEXT_ENWIKI_300.value, ft_language_code="da"):
        self.model_name = model_name
        self.ft_language_code = ft_language_code.lower()
        self.module_folder = Path(os.path.abspath(__file__)).parent.parent
        self.config_file = os.path.join(self.module_folder, "config", "cache_config.yml")
        self.cfg = utils.read_config(self.config_file)
        logging.getLogger(__name__)

    def get_wv_path(self):
        wv_nfo = self.get_wv_info()
        if wv_nfo["is_downloaded"] is False:
            self._download()
        return wv_nfo["wv_file_path"]

    def _get_data_folder(self):
        data_cache_path = str(self.cfg["data_cache_path"])
        if "{home}" in data_cache_path:
            data_cache_path = data_cache_path.format(home=Path.home())
        return Path(data_cache_path)

    def get_wv_list(self):
        return list(self.cfg["model_name"])

    def get_wv_info(self):
        root_key = "pretrained_vectors"

        cache_dir = self._get_data_folder()
        mdl_familly = self.cfg[root_key][self.model_name]["mdl_familly"]

        dl_src = self.cfg[root_key][self.model_name]["type_url"]
        dl_size = self.cfg[root_key][self.model_name]["size"]

        wv_folder = os.path.join(cache_dir, mdl_familly)
        wv_dim = self.cfg[root_key][self.model_name]["dim"]
        if self.model_name == WV.FASTTEXT_XXWIKI_300.value:
            wv_file_name = self.cfg[root_key][self.model_name]["wv_file"].format(
                self.ft_language_code
            )
            dl_url = self.cfg[root_key][self.model_name]["url"].format(
                self.ft_language_code
            )
        else:
            wv_file_name = self.cfg[root_key][self.model_name]["wv_file"]
            dl_url = self.cfg[root_key][self.model_name]["url"]
        wv_file_path = os.path.join(wv_folder, wv_file_name)
        is_downloaded = os.path.exists(wv_file_path)

        dic = {
            "mdl_name": self.model_name,
            "mdl_familly": mdl_familly,
            "is_downloaded": is_downloaded,
            "dl_url": dl_url,
            "dl_src": dl_src,
            "dl_size": dl_size,
            "wv_file_name": wv_file_name,
            "wv_folder": wv_folder,
            "wv_file_path": wv_file_path,
            "wv_dim": wv_dim,
        }
        return dic

    def _download(self, force_dl=False, delete_archive=True):

        wv_nfo = self.get_wv_info()

        dl_url = wv_nfo["dl_url"]
        dl_src = wv_nfo["dl_src"]
        dl_filename = os.path.basename(dl_url)
        dl_filename_ext = Path(dl_filename).suffix
        if dl_filename_ext in [".gz", ".zip", ".tar"]:
            is_archive = True
        else:
            is_archive = False
        dl_size = wv_nfo["dl_size"]
        dl_dest_folder = wv_nfo["wv_folder"]
        dl_dest_file = os.path.join(wv_nfo["wv_folder"], dl_filename)
        if "FASTTEXT_XX" in self.model_name:
            logging.info(
                "Downloading Pretrained vectors : {}...".format(
                    self.model_name.replace("XX", self.ft_language_code.upper())
                )
            )
        else:
            logging.info(
                "Downloading Pretrained vectors : {}...".format(self.model_name)
            )
        if force_dl or not os.path.exists(dl_dest_file):
            if not os.path.exists(dl_dest_folder):
                os.makedirs(dl_dest_folder)
            if dl_src == "gdrive":
                download_file_from_google_drive(dl_url, dl_dest_file, dl_size)
            else:
                download_from_url(dl_url, dl_dest_file)

            if is_archive:
                logging.info("Extracting Archive Data...")
                if dl_filename_ext == ".gz":
                    tar = tarfile.open(dl_dest_file, "r:gz")
                    for member in tar.getmembers():
                        if member.isreg():
                            member.name = os.path.basename(member.name)
                            tar.extract(member, dl_dest_folder)
                    tar.close()
                    logging.info("Extraction completed!")

                elif dl_filename_ext == ".tar":
                    tar = tarfile.open(dl_dest_file, "r:")
                    for member in tar.getmembers():
                        if member.isreg():
                            member.name = os.path.basename(member.name)
                            tar.extract(member, dl_dest_folder)
                    tar.close()
                    logging.info("Extraction completed!")

                elif dl_filename_ext == ".zip":
                    with zipfile.ZipFile(dl_dest_file, "r") as zip_ref:
                        zip_ref.extractall(dl_dest_folder)
                    logging.info("Extraction completed!")

                if delete_archive:
                    logging.info("Deleting Archived Data...")
                    os.remove(dl_dest_file)
                    logging.info("Archive deletion completed!")


class DSManager(object):
    """
    Class to automatically download & send back a dataset csv files
    """
    def __init__(self, dataset_name=DS.IMDB.value):
        self.dataset_name = dataset_name
        self.module_folder = Path(os.path.abspath(__file__)).parent.parent
        self.config_file = os.path.join(self.module_folder, "config", "cache_config.yml")
        self.cfg = utils.read_config(self.config_file)
        logging.getLogger(__name__)

    def get_partition_paths(self):
        ds_nfo = self._get_ds_info()
        if ds_nfo["is_downloaded"] is False:
            self._download()

        train_file = None
        test_file = None
        val_file = None

        if ds_nfo["train_file_path"] is not None:
            train_file = ds_nfo["train_file_path"]
        if ds_nfo["test_file_path"] is not None:
            test_file = ds_nfo["test_file_path"]
        if ds_nfo["val_file_path"] is not None:
            val_file = ds_nfo["val_file_path"]

        return train_file, test_file, val_file

    def _get_data_folder(self):
        data_cache_path = str(self.cfg["data_cache_path"])
        if "{home}" in data_cache_path:
            data_cache_path = data_cache_path.format(home=Path.home())
        return Path(data_cache_path)

    def get_ds_list(self):
        return list(self.cfg["datasets"])

    def _get_ds_info(self):
        root_key = "datasets"
        cache_dir = self._get_data_folder()

        url = self.cfg[root_key][self.dataset_name]["url"]
        src = self.cfg[root_key][self.dataset_name]["type_url"]
        dl_fname = self.cfg[root_key][self.dataset_name]["name"]

        fsize = self.cfg[root_key][self.dataset_name]["size"]
        filename = dl_fname.split(".")[0]
        dest_folder = os.path.join(cache_dir, filename)
        dest_file = os.path.join(dest_folder, dl_fname)
        header = self.cfg[root_key][self.dataset_name]["header"]
        if header == "None":
            header = None
        text_col = self.cfg[root_key][self.dataset_name]["text_col"]
        lbl_col = self.cfg[root_key][self.dataset_name]["lbl_col"]
        train_file_name = self.cfg[root_key][self.dataset_name]["train_file"]
        test_file_name = self.cfg[root_key][self.dataset_name]["test_file"]
        val_file_name = self.cfg[root_key][self.dataset_name]["val_file"]
        label_file_name = self.cfg[root_key][self.dataset_name]["label_file"]
        train_file_path = os.path.join(dest_folder, train_file_name)
        is_downloaded = os.path.exists(train_file_path)
        test_file_path = os.path.join(dest_folder, test_file_name)
        if val_file_name == "None":
            val_file_name = None
            val_file_path = None
        else:
            val_file_path = os.path.join(dest_folder, val_file_name)
        if label_file_name == "None":
            label_file_name = None
            label_file_path = None
        else:
            label_file_path = os.path.join(dest_folder, label_file_name)

        dic = {
            "dataset_name": self.dataset_name,
            "is_downloaded": is_downloaded,
            "dl_url": url,
            "dl_src": src,
            "dl_filename": dl_fname,
            "dl_size": fsize,
            "dl_name": filename,
            "dl_dest_folder": dest_folder,
            "dl_dest_file": dest_file,
            "header": header,
            "text_col": text_col,
            "lbl_col": lbl_col,
            "train_file_name": train_file_name,
            "test_file_name": test_file_name,
            "val_file_name": val_file_name,
            "label_file_name": label_file_name,
            "train_file_path": train_file_path,
            "test_file_path": test_file_path,
            "val_file_path": val_file_path,
            "label_file_path": label_file_path,
        }
        return dic

    def _download(self, dataset_name=None, force_dl=False, delete_archive=True):

        if dataset_name == "ALL":
            ds_list = self.get_ds_list()
            for i in range(len(ds_list)):
                self._download(ds_list[i], force_dl, delete_archive)
        else:
            if dataset_name is not None:
                self.dataset_name = dataset_name
            ds_nfo = self._get_ds_info()

            dl_url = ds_nfo["dl_url"]
            dl_src = ds_nfo["dl_src"]
            dl_filename = ds_nfo["dl_filename"]
            dl_filename_ext = Path(dl_filename).suffix
            if dl_filename_ext in [".gz", ".zip", ".tar"]:
                is_archive = True
            else:
                is_archive = False
            dl_size = ds_nfo["dl_size"]
            dl_dest_folder = ds_nfo["dl_dest_folder"]
            dl_dest_file = ds_nfo["dl_dest_file"]

            logging.info("Downloading Dataset : {}...".format(self.dataset_name))
            if force_dl or not os.path.exists(dl_dest_folder):

                if not os.path.exists(dl_dest_folder):
                    os.makedirs(dl_dest_folder)
                if dl_src == "gdrive":
                    download_file_from_google_drive(dl_url, dl_dest_file, dl_size)
                else:
                    download_from_url(dl_url, dl_dest_file)

                if is_archive:
                    logging.info("Extracting Archive Data...")
                    if dl_filename_ext == ".gz":
                        tar = tarfile.open(dl_dest_file, "r:gz")
                        for member in tar.getmembers():
                            if member.isreg():
                                member.name = os.path.basename(member.name)
                                tar.extract(member, dl_dest_folder)
                        tar.close()
                        logging.info("Extraction Completed!")

                    elif dl_filename_ext == ".tar":
                        tar = tarfile.open(dl_dest_file, "r:")
                        for member in tar.getmembers():
                            if member.isreg():
                                member.name = os.path.basename(member.name)
                                tar.extract(member, dl_dest_folder)
                        tar.close()
                        logging.info("Extraction Completed!")

                    elif dl_filename_ext == ".zip":
                        with zipfile.ZipFile(dl_dest_file, "r") as zip_ref:
                            zip_ref.extractall(dl_dest_folder)
                        logging.info("Extraction Completed!")

                    if delete_archive:
                        logging.info("Deleting Archived Data...")
                        os.remove(dl_dest_file)
                        logging.info("Archive Deletion Completed!")

                    self._apply_post_dl_reformating()

    def _apply_post_dl_reformating(self, dataset_name=DS.IMDB.value):
        if dataset_name is not None:
            self.dataset_name = dataset_name
        nfo = self._get_ds_info()
        text_col = nfo["text_col"]

        if dataset_name in [
            DS.AG_NEWS.value,
            DS.AMAZON_FULL.value,
            DS.AMAZON_POLARITY.value,
            DS.DBPEDIA.value,
            DS.SOGOU.value,
            DS.YAHOO.value,
        ]:
            logging.info("Reformating dataset : {}...".format(dataset_name))
            logging.info("Aggregating text columns all-together...")
            df_train = pd.read_csv(nfo["train_file_path"], header=nfo["header"])
            df_test = pd.read_csv(nfo["test_file_path"], header=nfo["header"])

            if dataset_name == DS.YAHOO.value:
                df_train[df_train.columns[text_col]] = (
                    df_train[df_train.columns[1]]
                    + " "
                    + df_train[df_train.columns[2]]
                    + " "
                    + df_test[df_test.columns[3]]
                )
                df_test[df_test.columns[text_col]] = (
                    df_test[df_test.columns[1]]
                    + " "
                    + df_test[df_test.columns[2]]
                    + " "
                    + df_test[df_test.columns[3]]
                )
            else:
                df_train[df_train.columns[text_col]] = (
                    df_train[df_train.columns[1]] + " " + df_train[df_train.columns[2]]
                )
                df_test[df_test.columns[text_col]] = (
                    df_test[df_test.columns[1]] + " " + df_test[df_test.columns[2]]
                )

            df_train.to_csv(
                nfo["train_file_path"],
                header=nfo["header"],
                sep=",",
                encoding="utf8",
                index=False,
            )
            df_test.to_csv(
                nfo["test_file_path"],
                header=nfo["header"],
                sep=",",
                encoding="utf8",
                index=False,
            )

            logging.info("Aggregation of text columns completed!")
