import logging
import time
from pprint import pprint
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from nlplay.data.cache import DSManager, DS, WordVectorsManager, WV
from nlplay.features.fasttext_features import read_word_vectors
from nlplay.features.sentence_embeddings import USIFVectorizer
from nlplay.features.text_cleaner import base_cleaner
from nlplay.utils.parlib import parallelApply
from nlplay.utils.utils import get_elapsed_time

logging.basicConfig(
    format="%(asctime)s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

if __name__ == "__main__":

    train_csv, test_csv, val_csv = DSManager(DS.IMDB.value).get_partition_paths()
    pretrained_vec = WordVectorsManager(WV.GLOVE_EN6B_300.value).get_wv_path()

    # Inputs & Model Parameters
    m = 5  # number of common discourse vectors removed
    normalize = "reference"  # word vectors normalization of the uSIF reference code
    C_grid = np.logspace(-3, 2, 6).tolist()

    # Data preparation, base_cleaner lowercases and splits the punctuation as in GloVe 6B
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    X_train = parallelApply(df_train[df_train.columns[0]], base_cleaner, 3).tolist()
    X_test = parallelApply(df_test[df_test.columns[0]], base_cleaner, 3).tolist()
    y_train = df_train[df_train.columns[1]].to_numpy(copy=True)
    y_test = df_test[df_test.columns[1]].to_numpy(copy=True)

    start_time = time.time()
    words, vectors = read_word_vectors(pretrained_vec)
    word_vectors = dict(zip(words, vectors))
    logging.info(
        "{} word vectors loaded - Time elapsed: {}".format(
            len(word_vectors), get_elapsed_time(start_time)
        )
    )

    # uSIF sentence embeddings, unsupervised: word probabilities and common components
    # estimated on the train texts, n is the average text length (about 11 for STS)
    n = int(round(np.mean([len(x.split()) for x in X_train])))
    start_time = time.time()
    usif = USIFVectorizer(word_vectors, n=n, m=m, normalize=normalize)
    E_train = usif.fit_transform(X_train)
    E_test = usif.transform(X_test)
    logging.info(
        "uSIF embeddings (n={}, a={:.2e}) - Time elapsed: {}".format(
            n, usif.a_, get_elapsed_time(start_time)
        )
    )
    del word_vectors, vectors

    # Classifier tuned with a cross validation on the train embeddings
    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]
    )
    search = GridSearchCV(
        pipeline,
        {"clf__C": C_grid},
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy",
        n_jobs=-1,
    )
    search.fit(E_train, y_train)

    print("Best Parameters:")
    pprint(search.best_params_)
    print("CV accuracy : " + str(search.best_score_))
    print("Training accuracy : " + str(search.score(E_train, y_train)))

    # Score with the test data
    y_preds = search.predict(E_test)
    print("Test accuracy : " + str(accuracy_score(y_test, y_preds)))
    # 2026-09-25 08:54:23 400000 word vectors loaded - Time elapsed: 0m 11s
    # 2026-09-25 08:54:41 uSIF embeddings (n=274, a=5.88e-02) - Time elapsed: 0m 17s
    # Best Parameters:
    # {'clf__C': 0.01}
    # CV accuracy : 0.83944
    # Training accuracy : 0.84592
    # Test accuracy : 0.83672
