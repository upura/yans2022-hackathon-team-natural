import warnings
from itertools import combinations

# from kaggler.preprocessing import TargetEncoder
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import normalized_mutual_info_score
from tqdm import tqdm

from .base import *
from .category_embedding import *
from .category_encoding import *
from .dae import *
from .graph import *
from .groupby import *
from .image import *
from .image_pretrained import *
from .row_aggregations import *
from .selection import *
from .text import *


def merge_columns(dataframe, columns):
    new_column = "_".join(columns)
    dataframe[new_column] = ""
    for c in columns:
        dataframe[new_column] += dataframe[c].astype(str).fillna("null")

    return dataframe


def merge_columns_with_mutual_info_score(dataframe, columns, threshold=0.3):
    for c1, c2 in combinations(columns, 2):
        if (
            normalized_mutual_info_score(
                dataframe[c1], dataframe[c2], average_method="arithmetic"
            )
            > threshold
        ):
            dataframe = merge_columns(dataframe, [c1, c2])
    return dataframe


def get_interactions(dataframe, interaction_features):
    for (c1, c2) in combinations(interaction_features, 2):
        dataframe[c1 + "_mul_" + c2] = dataframe[c1] * dataframe[c2]
        dataframe[c1 + "_div_" + c2] = dataframe[c1] / dataframe[c2]
    return dataframe


def get_time_features(dataframe, time_column):
    dataframe[time_column] = pd.to_datetime(dataframe[time_column])
    dataframe[time_column + "_day"] = dataframe[time_column].dt.day
    dataframe[time_column + "_dayofweek"] = dataframe[time_column].dt.dayofweek
    dataframe[time_column + "_weekofyear"] = dataframe[time_column].dt.weekofyear
    dataframe[time_column + "_weekofmonth"] = dataframe[time_column].dt.day // 7
    dataframe[time_column + "_is_weekend"] = (
        dataframe[time_column].dt.weekday >= 5
    ).astype(np.uint8)
    dataframe[time_column + "_month"] = dataframe[time_column].dt.month
    dataframe[time_column + "_year"] = dataframe[time_column].dt.year
    dataframe[time_column + "_hour"] = dataframe[time_column].dt.hour
    dataframe[time_column + "_minute"] = dataframe[time_column].dt.minute
    dataframe[time_column + "_second"] = dataframe[time_column].dt.second
    return dataframe


def count_null(train: pd.DataFrame, col_definition):
    train["count_null"] = train.isnull().sum(axis=1)
    for f in col_definition:
        if sum(train[f].isnull().astype(int)) > 0:
            train[f"cn_{f}"] = train[f].isnull().astype(int)
    return train.loc[:, train.columns.str.contains("cn_")]


def count_encoding(train: pd.DataFrame, col_definition):
    for f in col_definition:
        count_map = train[f].value_counts().to_dict()
        train[f"ce_{f}"] = train[f].map(count_map)
    return train.loc[:, train.columns.str.contains("ce_")]


def count_encoding_interact(train: pd.DataFrame, col_definition):
    for col1, col2 in tqdm(list(combinations(col_definition, 2))):
        col = col1 + "_" + col2
        _tmp = train[col1].astype(str) + "_" + train[col2].astype(str)
        count_map = _tmp.value_counts().to_dict()
        train[f"cei_{col}"] = _tmp.map(count_map)

    return train.loc[:, train.columns.str.contains("cei_")]


def target_encoding(
    train: pd.DataFrame, test: pd.DataFrame, encode_col, target_col, cv
):
    warnings.simplefilter("ignore")
    te = TargetEncoder(cv=cv)

    train_fe = te.fit_transform(train[encode_col], train[target_col])
    train_fe.columns = ["te_" + c for c in train_fe.columns]

    test_fe = te.transform(test[encode_col])
    test_fe.columns = ["te_" + c for c in test_fe.columns]

    return pd.concat([train_fe, test_fe]).reset_index(drop=True)


def matrix_factorization(train: pd.DataFrame, col_definition, option):
    """
    col_definition: encode_col
    option: n_components_lda, n_components_svd
    """

    cf = CategoryVectorizer(
        col_definition,
        option["n_components_lda"],
        vectorizer=CountVectorizer(),
        transformer=LatentDirichletAllocation(
            n_components=option["n_components_lda"],
            n_jobs=-1,
            learning_method="online",
            random_state=777,
        ),
        name="CountLDA",
    )
    features_lda = cf.transform(train).astype(np.float32)

    cf = CategoryVectorizer(
        col_definition,
        option["n_components_svd"],
        vectorizer=CountVectorizer(),
        transformer=TruncatedSVD(
            n_components=option["n_components_svd"], random_state=777
        ),
        name="CountSVD",
    )
    features_svd = cf.transform(train).astype(np.float32)

    return features_svd, features_lda
