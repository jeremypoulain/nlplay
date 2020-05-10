import string
import re
from blingfire import text_to_words
from bs4 import BeautifulSoup
from nlplay.features.text_expressions import *


def blingf_tokenizer(s: str):
    return text_to_words(s)


def replace_non_alphanumeric(text: str, replace_with=" "):
    return NON_ALPHA_NUMERIC.sub(replace_with, text)


def replace_multi_whitespaces(text: str, replace_with=" "):
    return MULTI_WHITESPACE_REGEX.sub(replace_with, text)


def replace_control_chars(text: str):
    return text.translate(CONTROL_CHARS)


def replace_line_breaks(text: str, replace_with=" "):
    return LINEBREAK_REGEX.sub(replace_with, text)


def replace_single_quotes(text: str, replace_with="'"):
    return SINGLE_QUOTE_REGEX.sub(replace_with, text)


def replace_double_quotes(text: str, replace_with='"'):
    return DOUBLE_QUOTE_REGEX.sub(replace_with, text)


def normalize_all_quotes(text: str):
    return replace_single_quotes(replace_double_quotes(text))


def replace_en_stop_words(text: str, replace_with=" "):
    return EN_STOP_WORDS_REGEX.sub(replace_with, text)


def replace_currencies(text: str, replace_with=" currency "):
    return NON_ALPHA_NUMERIC.sub(replace_with, text)


def replace_emails(text: str, replace_with=" email "):
    return EMAIL_REGEX.sub(replace_with, text)


def replace_html(text: str):
    return BeautifulSoup(text, features="html.parser").get_text()


def replace_en_contracted_forms_quick(text: str):
    # specific
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)

    # general
    text = re.sub(r"n\'t ", " not ", text)
    text = re.sub(r"\'t ", " not ", text)
    text = re.sub(r"\'re ", " are ", text)
    text = re.sub(r"\'s ", " is ", text)
    text = re.sub(r"\'d ", " would ", text)
    text = re.sub(r"\'ll ", " will ", text)
    text = re.sub(r"\'ve ", " have ", text)
    text = re.sub(r"\'m ", " am ", text)

    return text


def replace_en_contracted_forms_full(text: str):
    def replace(match):
        return en_contractions[match.group(0)]
    return EN_CONTRACTIONS_REGEX.sub(replace, text)


def base_cleaner(text: str):
    text = text.lower()
    text = normalize_all_quotes(text)
    text = replace_en_contracted_forms_quick(text)
    text = replace_html(text)
    text = blingf_tokenizer(text)
    return text


def base_aggresive_cleaner(text: str):
    text = text.lower()
    text = replace_en_stop_words(text)
    text = replace_control_chars(text)
    text = replace_en_contracted_forms_quick(text)
    text = replace_html(text)
    text = replace_non_alphanumeric(text)
    text = blingf_tokenizer(text)
    return text


def ft_cleaner(text: str):
    """
    text cleaning function used in FastText Paper
    :param text:
    :return:
    """
    text = text.lower()
    text = text.replace("'", " ' ").replace(".", " . ").replace("!", " ! ").replace(",", " , ")
    text = text.replace("(", " ( ").replace(")", " ) ").replace("?", " ? ")
    text = text.replace('"', "").replace(";", "").replace(":", "")
    text = replace_html(text)
    return text


def kimyoon_text_cleaner(text: str):
    """
    Kim yoon text cleaning function used in its CNN paper
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
    text = re.sub(r"\'s", " \'s", text)
    text = re.sub(r"\'ve", " \'ve", text)
    text = re.sub(r"n\'t", " n\'t", text)
    text = re.sub(r"\'re", " \'re", text)
    text = re.sub(r"\'d", " \'d", text)
    text = re.sub(r"\'ll", " \'ll", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip().lower()

