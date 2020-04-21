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
    FASTTEXT_XXWIKI_300 = 'FASTTEXT_XXWIKI_300'
    FASTTEXT_ENWIKI_300 = 'FASTTEXT_ENWIKI_300'
    FASTTEXT_ENWIKI_SUB_300 = 'FASTTEXT_ENWIKI_SUB_300'
    FASTTEXT_ENCC_300 = 'FASTTEXT_ENCC_300'
    FASTTEXT_ENCC_SUB_300 = 'FASTTEXT_ENCC_SUB_300'
    GLOVE_EN6B_50 = 'GLOVE_EN6B_50'
    GLOVE_EN6B_100 = 'GLOVE_EN6B_100'
    GLOVE_EN6B_200 = 'GLOVE_EN6B_200'
    GLOVE_EN6B_300 = 'GLOVE_EN6B_300'
    GLOVE_EN840B_300 = 'GLOVE_EN840B_300'


class DS(Enum):
    def __str__(self):
        return str(self.value)
    TWENTY_NEWS = '20_NEWS'
    AG_NEWS = 'AG_NEWS'
    AMAZON_FULL = 'AMAZON_FULL'
    AMAZON_POLARITY = 'AMAZON_POLARITY'
    DBPEDIA = 'DBPEDIA'
    IMDB = 'IMDB'
    MR = 'MR'
    SOGOU = 'SOGOU'
    SST2 = 'SST2'
    TREC_6 = 'TREC_6'
    TREC_50 = 'TREC_50'
    YAHOO = 'YAHOO'
    YELP_FULL = 'YELP_FULL'
    YELP_POLARITY = 'YELP_POLARITY'


class WordVectors(object):
    def __init__(self, config_file):
        self.model_name = None
        self.config_file = config_file
        self.cfg = utils.read_config(self.config_file)
        logging.getLogger(__name__)

    def get_root_folder(self):
        return str(Path(self.cfg['cache']).resolve())

    def get_wv_list(self):
        return list(self.cfg['model_name'])

    def get_wv_info(self, model_name=WV.FASTTEXT_ENWIKI_300.value, ft_language_code='da', property_keys=None):
        root_key = 'pretrained_vectors'

        self.model_name = model_name
        cache_dir = self.get_root_folder()
        mdl_familly = self.cfg[root_key][self.model_name]['mdl_familly']

        dl_src = self.cfg[root_key][self.model_name]['type_url']
        dl_size = self.cfg[root_key][self.model_name]['size']

        wv_folder = os.path.join(cache_dir, mdl_familly)
        wv_dim = self.cfg[root_key][self.model_name]['dim']
        if model_name == WV.FASTTEXT_XXWIKI_300.value:
            wv_file_name = self.cfg[root_key][self.model_name]['wv_file'].format(ft_language_code)
            dl_url = self.cfg[root_key][self.model_name]['url'].format(ft_language_code)
        else:
            wv_file_name = self.cfg[root_key][self.model_name]['wv_file']
            dl_url = self.cfg[root_key][self.model_name]['url']
        wv_file_path = os.path.join(wv_folder, wv_file_name)
        is_downloaded = os.path.exists(wv_file_path)

        dic = {'mdl_name': self.model_name,
               'mdl_familly': mdl_familly,
               'is_downloaded': is_downloaded,
               'dl_url': dl_url,
               'dl_src': dl_src,
               'dl_size': dl_size,
               'wv_file_name': wv_file_name,
               'wv_folder': wv_folder,
               'wv_file_path': wv_file_path,
               'wv_dim': wv_dim,
               }
        return dic

    def download(self, model_name=WV.GLOVE_EN6B_50.value, ft_language_code='da',
                 force_dl=False, delete_archive=False):
        self.model_name = model_name
        nfo = self.get_wv_info(model_name, ft_language_code)

        dl_url = nfo['dl_url']
        dl_src = nfo['dl_src']
        dl_filename = os.path.basename(dl_url)
        dl_filename_ext = Path(dl_filename).suffix
        if dl_filename_ext in ['.gz', '.zip', '.tar']:
            is_archive = True
        else:
            is_archive = False
        dl_size = nfo['dl_size']
        dl_dest_folder = nfo['wv_folder']
        dl_dest_file = os.path.join(nfo['wv_folder'], dl_filename)

        logging.info('Downloading Pretrained vectors : {}...'.format(model_name))
        if force_dl or not os.path.exists(dl_dest_file):

            if not os.path.exists(dl_dest_folder):
                os.makedirs(dl_dest_folder)
            if dl_src == 'gdrive':
                download_file_from_google_drive(dl_url, dl_dest_file, dl_size)
            else:
                download_from_url(dl_url, dl_dest_file)

            if is_archive:
                logging.info('Extracting Archive Data...')
                if dl_filename_ext == ".gz":
                    tar = tarfile.open(dl_dest_file, "r:gz")
                    for member in tar.getmembers():
                        if member.isreg():
                            member.name = os.path.basename(member.name)
                            tar.extract(member, dl_dest_folder)
                    tar.close()
                    logging.info('Extraction completed!')

                elif dl_filename_ext == ".tar":
                    tar = tarfile.open(dl_dest_file, "r:")
                    for member in tar.getmembers():
                        if member.isreg():
                            member.name = os.path.basename(member.name)
                            tar.extract(member, dl_dest_folder)
                    tar.close()
                    logging.info('Extraction completed!')

                elif dl_filename_ext == ".zip":
                    with zipfile.ZipFile(dl_dest_file, "r") as zip_ref:
                        zip_ref.extractall(dl_dest_folder)
                    logging.info('Extraction completed!')

                if delete_archive:
                    logging.info('Deleting Archived Data...')
                    os.remove(dl_dest_file)
                    logging.info('Archive deletion completed!')


class Datasets(object):
    def __init__(self, config_file):
        self.dataset_name = None
        self.config_file = config_file
        self.cfg = utils.read_config(self.config_file)
        logging.getLogger(__name__)

    def get_root_folder(self):
        return str(Path(self.cfg['cache']).resolve())

    def get_ds_list(self):
        return list(self.cfg['datasets'])

    def get_ds_info(self, dataset_name=DS.IMDB.value, keys=None):
        root_key = 'datasets'

        self.dataset_name = dataset_name
        cache_dir = self.get_root_folder()

        url = self.cfg[root_key][self.dataset_name]['url']
        src = self.cfg[root_key][self.dataset_name]['type_url']
        dl_fname = self.cfg[root_key][self.dataset_name]['name']

        fsize = self.cfg[root_key][self.dataset_name]['size']
        filename = dl_fname.split('.')[0]
        dest_folder = os.path.join(cache_dir, filename)
        dest_file = os.path.join(dest_folder, dl_fname)
        header = self.cfg[root_key][self.dataset_name]['header']
        if header == 'None':
            header = None
        text_col = self.cfg[root_key][self.dataset_name]['text_col']
        lbl_col = self.cfg[root_key][self.dataset_name]['lbl_col']
        train_file_name = self.cfg[root_key][self.dataset_name]['train_file']
        test_file_name = self.cfg[root_key][self.dataset_name]['test_file']
        val_file_name = self.cfg[root_key][self.dataset_name]['val_file']
        label_file_name = self.cfg[root_key][self.dataset_name]['label_file']
        train_file_path = os.path.join(dest_folder, train_file_name)
        is_downloaded = os.path.exists(train_file_path)
        test_file_path = os.path.join(dest_folder, test_file_name)
        if val_file_name == 'None':
            val_file_name = None
            val_file_path = None
        else:
            val_file_path = os.path.join(dest_folder, val_file_name)
        if label_file_name == 'None':
            label_file_name = None
            label_file_path = None
        else:
            label_file_path = os.path.join(dest_folder, label_file_name)

        dic = {'dataset_name': dataset_name,
               'is_downloaded': is_downloaded,
               'dl_url': url,
               'dl_src': src,
               'dl_filename': dl_fname,
               'dl_size': fsize,
               'dl_name': filename,
               'dl_dest_folder': dest_folder,
               'dl_dest_file': dest_file,
               'header': header,
               'text_col': text_col,
               'lbl_col': lbl_col,
               'train_file_name': train_file_name,
               'test_file_name': test_file_name,
               'val_file_name': val_file_name,
               'label_file_name': label_file_name,
               'train_file_path': train_file_path,
               'test_file_path': test_file_path,
               'val_file_path': val_file_path,
               'label_file_path': label_file_path
               }
        return dic

    def download(self, dataset_name=DS.IMDB.value, force_dl=False, delete_archive=False):

        if dataset_name == 'ALL':
            ds_list = self.get_ds_list()
            for i in range(len(ds_list)):
                self.download(ds_list[i], force_dl, delete_archive)
        else:
            self.dataset_name = dataset_name
            nfo = self.get_ds_info(dataset_name)

            dl_url = nfo['dl_url']
            dl_src = nfo['dl_src']
            dl_filename = nfo['dl_filename']
            dl_filename_ext = Path(dl_filename).suffix
            if dl_filename_ext in ['.gz', '.zip', '.tar']:
                is_archive = True
            else:
                is_archive = False
            dl_size = nfo['dl_size']
            dl_dest_folder = nfo['dl_dest_folder']
            dl_dest_file = nfo['dl_dest_file']

            logging.info('Downloading Dataset : {}...'.format(dataset_name))
            if force_dl or not os.path.exists(dl_dest_folder):

                if not os.path.exists(dl_dest_folder):
                    os.makedirs(dl_dest_folder)
                if dl_src == 'gdrive':
                    download_file_from_google_drive(dl_url, dl_dest_file, dl_size)
                else:
                    download_from_url(dl_url, dl_dest_file)

                if is_archive:
                    logging.info('Extracting Archive Data...')
                    if dl_filename_ext == ".gz":
                        tar = tarfile.open(dl_dest_file, "r:gz")
                        for member in tar.getmembers():
                            if member.isreg():
                                member.name = os.path.basename(member.name)
                                tar.extract(member, dl_dest_folder)
                        tar.close()
                        logging.info('Extraction Completed!')

                    elif dl_filename_ext == ".tar":
                        tar = tarfile.open(dl_dest_file, "r:")
                        for member in tar.getmembers():
                            if member.isreg():
                                member.name = os.path.basename(member.name)
                                tar.extract(member, dl_dest_folder)
                        tar.close()
                        logging.info('Extraction Completed!')

                    elif dl_filename_ext == ".zip":
                        with zipfile.ZipFile(dl_dest_file, "r") as zip_ref:
                            zip_ref.extractall(dl_dest_folder)
                        logging.info('Extraction Completed!')

                    if delete_archive:
                        logging.info('Deleting Archived Data...')
                        os.remove(dl_dest_file)
                        logging.info('Archive Deletion Completed!')

                    self.apply_post_dl_reformating(dataset_name)

    def apply_post_dl_reformating(self, dataset_name=DS.IMDB.value):

        nfo = self.get_ds_info(dataset_name)
        text_col = nfo['text_col']

        if dataset_name in [DS.AG_NEWS.value, DS.AMAZON_FULL.value, DS.AMAZON_POLARITY.value, DS.DBPEDIA.value,
                            DS.SOGOU.value, DS.YAHOO.value]:
            logging.info('Reformating dataset : {}...'.format(dataset_name))
            logging.info('Aggregating text columns all-together...')
            df_train = pd.read_csv(nfo['train_file_path'], header=nfo['header'])
            df_test = pd.read_csv(nfo['test_file_path'], header=nfo['header'])

            if dataset_name == DS.YAHOO.value:
                df_train[df_train.columns[text_col]] = df_train[df_train.columns[1]] + ' ' + \
                                                       df_train[df_train.columns[2]] + ' ' + \
                                                       df_test[df_test.columns[3]]
                df_test[df_test.columns[text_col]] = df_test[df_test.columns[1]] + ' ' + \
                                                     df_test[df_test.columns[2]] + ' ' + \
                                                     df_test[df_test.columns[3]]
            else:
                df_train[df_train.columns[text_col]] = df_train[df_train.columns[1]] + ' ' + \
                                                       df_train[df_train.columns[2]]
                df_test[df_test.columns[text_col]] = df_test[df_test.columns[1]] + ' ' + \
                                                     df_test[df_test.columns[2]]

            df_train.to_csv(nfo['train_file_path'], header=nfo['header'], sep=',', encoding='utf8', index=False)
            df_test.to_csv(nfo['test_file_path'], header=nfo['header'], sep=',', encoding='utf8', index=False)

            logging.info('Aggregation of text columns completed!')


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG, datefmt="%Y-%m-%d %H:%M:%S")
    yml = 'cache_config.yml'
    obj = Datasets(yml)
    # for dataset_name in [DS.IMDB.value]:
    #     obj.download(dataset_name)
    #     obj.apply_post_dl_reformating(dataset_name)

    obj = WordVectors(yml)
    obj.download(WV.GLOVE_EN6B_100.value)