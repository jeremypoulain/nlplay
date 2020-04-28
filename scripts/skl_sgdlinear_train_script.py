import re
import string
from pprint import pprint
import numpy as np
import pandas as pd
from hyperopt import fmin, tpe, Trials, space_eval, hp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from nlplay.features.text_cleaner import base_cleaner
from nlplay.utils.parlib import parallelApply

if __name__ == "__main__":

    train_csv = "../nlplay/data_cache/IMDB/IMDB_train.csv"
    test_csv = "../nlplay/data_cache//IMDB/IMDB_test.csv"

    # Data preparation
    df_train = pd.read_csv(train_csv)
    df_train = df_train.sample(frac=1)
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

    re_tok = re.compile("([%s“”¨«»®´·º½¾¿¡§£₤‘’])" % string.punctuation)
    tokenizer = lambda x: re_tok.sub(r" \1 ", x).split()

    # Pipeline definition
    pipeline = Pipeline(
        [("vect", TfidfVectorizer(sublinear_tf=True)), ("clf", SGDClassifier(loss="modified_huber"))]
    )

    # Parameter search space
    space = {}
    space["vect__ngram_range"] = hp.choice("vect__ngram_range", [(1, 2), (1, 3)])
    space["vect__min_df"] = 1 + hp.randint("vect__min_df", 5)
    space["vect__max_df"] = hp.uniform("vect__max_df", 0.80, 1.0)
    space["clf__alpha"] = hp.loguniform("clf__alpha", -9 * np.log(10), -4 * np.log(10))

    # Define Hyperopt objective function - ie we want to maximize accuracy
    def objective(params):
        pipeline.set_params(**params)
        shuffle = KFold(n_splits=5, shuffle=True)
        score = cross_val_score(
            pipeline, X_train, y_train, cv=shuffle, scoring="accuracy", n_jobs=-1
        )
        return 1 - score.mean()

    # The Trials object will store details of each iteration
    trials = Trials()

    # Run hyperparameter search using the tpe algorithm
    best = fmin(objective, space, algo=tpe.suggest, max_evals=15, trials=trials)

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
    # 100%|██████████| 15/15 [06:42<00:00, 26.83s/trial, best loss: 0.09144000000000008]
    # Best Parameters:
    # {'clf__alpha': 2.6877296252886694e-05,
    #  'vect__max_df': 0.8482243048758884,
    #  'vect__min_df': 1,
    #  'vect__ngram_range': (1, 2)}
    # Training accuracy : 1.0
    # Test accuracy : 0.90696