import re
import string
from pprint import pprint
import numpy as np
import pandas as pd
from hyperopt import fmin, tpe, Trials, space_eval, hp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from nlplay.data.cache import DSManager, DS
from nlplay.features.text_cleaner import base_cleaner
from nlplay.models.sklearn.classifiers.skl_nbsvm import NBSVM
from nlplay.utils.parlib import parallelApply

re_tok = re.compile("([%s“”¨«»®´·º½¾¿¡§£₤‘’])" % string.punctuation)


def tokenizer(text):
    # Punctuation kept as tokens, as in the NBSVM paper
    return re_tok.sub(r" \1 ", text).split()


if __name__ == "__main__":

    train_csv, test_csv, val_csv = DSManager(DS.IMDB.value).get_partition_paths()

    # Data preparation
    df_train = pd.read_csv(train_csv)
    df_train = df_train.sample(frac=1, random_state=42)
    df_test = pd.read_csv(test_csv)
    df_train[df_train.columns[0]] = parallelApply(
        df_train[df_train.columns[0]], base_cleaner, 3
    )
    df_test[df_test.columns[0]] = parallelApply(
        df_test[df_test.columns[0]], base_cleaner, 3
    )

    # Train/test set creation
    X_train = df_train[df_train.columns[0]].tolist()
    y_train = df_train[df_train.columns[1]].tolist()
    X_test = df_test[df_test.columns[0]].tolist()
    y_test = df_test[df_test.columns[1]].tolist()

    # Pipeline definition, NBSVM binarizes the counts (binarize=True) as in the paper
    pipeline = Pipeline(
        [
            ("vect", CountVectorizer(tokenizer=tokenizer, token_pattern=None)),
            ("clf", NBSVM(random_state=42)),
        ]
    )

    # Parameter search space
    space = {}
    space["vect__ngram_range"] = hp.choice("vect__ngram_range", [(1, 2), (1, 3)])
    space["vect__min_df"] = 1 + hp.randint("vect__min_df", 3)
    space["clf__loss"] = hp.choice("clf__loss", ["log_loss", "hinge", "modified_huber"])
    space["clf__alpha"] = hp.uniform("clf__alpha", 0.1, 1.0)
    space["clf__sgd_alpha"] = hp.loguniform(
        "clf__sgd_alpha", -8 * np.log(10), -3 * np.log(10)
    )

    # Define Hyperopt objective function - ie we want to maximize accuracy
    def objective(params):
        pipeline.set_params(**params)
        shuffle = KFold(n_splits=5, shuffle=True, random_state=42)
        score = cross_val_score(
            pipeline, X_train, y_train, cv=shuffle, scoring="accuracy", n_jobs=-1
        )
        return 1 - score.mean()

    # The Trials object will store details of each iteration
    trials = Trials()

    # Run hyperparameter search using the tpe algorithm
    best = fmin(
        objective,
        space,
        algo=tpe.suggest,
        max_evals=15,
        trials=trials,
        rstate=np.random.default_rng(42),
    )

    # Get the values of the optimal parameters
    best_params = space_eval(space, best)

    print("Best Parameters:")
    pprint(best_params)

    # Fit the model with the optimal hyperparameters
    pipeline.set_params(**best_params)
    pipeline.fit(X_train, y_train)
    print("Training accuracy : " + str(pipeline.score(X_train, y_train)))

    # Score with the test data
    y_preds = pipeline.predict(X_test)
    print("Test accuracy : " + str(accuracy_score(y_test, y_preds)))
    # 100%|██████████| 15/15 [03:48<00:00, 15.23s/trial, best loss: 0.0809200000000001]
    # Best Parameters:
    # {'clf__alpha': 0.8824203246567753,
    #  'clf__loss': 'log_loss',
    #  'clf__sgd_alpha': 0.0009659086370538275,
    #  'vect__min_df': 3,
    #  'vect__ngram_range': (1, 3)}
    # Training accuracy : 0.9962
    # Test accuracy : 0.918
