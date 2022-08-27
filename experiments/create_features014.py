import sys

sys.path.append("../")

import os
from typing import Any

import joblib
import MeCab
import pandas as pd
from kaggle_utils.features import count_encoding, count_encoding_interact
from kaggle_utils.features.category_encoding import CategoricalEncoder
from kaggle_utils.features.groupby import (
    DiffGroupbyTransformer,
    GroupbyTransformer,
    RatioGroupbyTransformer,
)
from kaggle_utils.features.text import BasicTextFeatureTransformer, TextVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureStore:
    def __init__(self, feature_names: str, target_col: str) -> None:
        self.feature_names = feature_names
        self.target_col = target_col
        _res = pd.concat([pd.read_feather(f) for f in feature_names], axis=1)

        _train = _res.dropna(subset=[target_col]).copy()
        _test = _res.loc[_res[target_col].isnull()].copy()

        self.X_train = _train.drop(target_col, axis=1)
        self.y_train = _train[target_col]
        self.X_test = _test.drop(target_col, axis=1)


class Data:
    @classmethod
    def dump(cls, value: Any, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(value, path, compress=True)

    @classmethod
    def load(cls, path: str) -> Any:
        return joblib.load(path)


if __name__ == "__main__":
    train = pd.read_json(
        "../input/yans2022/training.jsonl", orient="records", lines=True
    )
    test = pd.read_json(
        "../input/yans2022/leader_board.jsonl", orient="records", lines=True
    )
    final = pd.read_json(
        "../input/yans2022/final_result.jsonl", orient="records", lines=True
    )
    train_test = pd.concat([train, test, final], axis=0).reset_index(drop=True)
    print(train.shape, test.shape, final.shape)

    categorical_cols = [
        "product_category",
        "vine",
        "verified_purchase",
        "product_idx",
        "customer_idx",
    ]
    text_cols = ["product_title", "review_headline", "review_body"]
    target_col = ["helpful_votes"]

    train_test["review_date"] = pd.to_datetime(train_test["review_date"])
    train_test = pd.merge(
        train_test,
        train_test.groupby("product_idx")["review_date"].min().reset_index(),
        on="product_idx",
        how="left",
        suffixes=("", "_first"),
    )
    train_test["review_days"] = (
        train_test["review_date"] - train_test["review_date_first"]
    ).dt.days

    train_test["review_year"] = train_test["review_date"].dt.year
    train_test["review_month"] = train_test["review_date"].dt.month
    numerical_cols = ["star_rating", "review_days", "review_year", "review_month"]

    # label_encoding
    ce = CategoricalEncoder(categorical_cols)
    train_test = ce.transform(train_test)
    train_test[categorical_cols + numerical_cols + target_col].to_feather(
        "../input/feather/train_test.ftr"
    )

    # count_encoding
    count_encoding(train_test, categorical_cols).to_feather(
        "../input/feather/count_encoding.ftr"
    )

    # count_encoding_interact
    count_encoding_interact(train_test, categorical_cols).to_feather(
        "../input/feather/count_encoding_interact.ftr"
    )

    path = "-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd"
    m = MeCab.Tagger(path)

    # text
    train_test["review_headline"] = [
        [line.split()[0] for line in m.parse(cp).splitlines()]
        for cp in train_test["review_headline"]
    ]
    train_test["review_body"] = [
        [line.split()[0] for line in m.parse(cp).splitlines() if len(line.split())]
        for cp in train_test["review_body"]
    ]
    train_test["product_title"] = [
        [line.split()[0] for line in m.parse(cp).splitlines() if len(line.split())]
        for cp in train_test["product_title"]
    ]

    train_test["review_headline"] = [
        " ".join(text) for text in train_test["review_headline"]
    ]
    train_test["review_body"] = [" ".join(text) for text in train_test["review_body"]]
    train_test["product_title"] = [
        " ".join(text) for text in train_test["product_title"]
    ]

    before_cols = train_test.columns
    btft = BasicTextFeatureTransformer(text_columns=["review_headline", "review_body"])
    train_test = btft.transform(train_test)

    tcv = TextVectorizer(text_columns=["review_headline"], name="review_headline_count")
    train_test = tcv.transform(train_test)
    scv = TextVectorizer(text_columns=["review_body"], name="review_body_count")
    train_test = scv.transform(train_test)
    pcv = TextVectorizer(text_columns=["product_title"], name="product_title_count")
    train_test = pcv.transform(train_test)

    ttv = TextVectorizer(
        text_columns=["review_headline"],
        vectorizer=TfidfVectorizer(),
        transformer=TruncatedSVD(n_components=128, random_state=777),
        name="review_headline_tfidf",
    )
    train_test = ttv.transform(train_test)
    stv = TextVectorizer(
        text_columns=["review_body"],
        vectorizer=TfidfVectorizer(),
        transformer=TruncatedSVD(n_components=128, random_state=777),
        name="review_body_tfidf",
    )
    train_test = stv.transform(train_test)
    ptv = TextVectorizer(
        text_columns=["product_title"],
        vectorizer=TfidfVectorizer(),
        transformer=TruncatedSVD(n_components=128, random_state=777),
        name="product_title_tfidf",
    )
    train_test = ptv.transform(train_test)

    after_cols = train_test.columns
    train_test[list(set(after_cols) - set(before_cols))].to_feather(
        "../input/feather/bow_tfidf.ftr"
    )

    # aggregation
    groupby_dict = [
        {
            "key": ["customer_idx"],
            "var": ["review_days", "ce_product_idx"],
            "agg": ["mean", "sum", "median", "min", "max", "var", "std"],
        },
        {
            "key": ["product_category"],
            "var": ["review_days", "ce_product_idx"],
            "agg": ["mean", "sum", "median", "min", "max", "var", "std"],
        },
        {
            "key": ["product_idx"],
            "var": ["review_days"],
            "agg": ["mean", "sum", "median", "min", "max", "var", "std"],
        },
        {
            "key": ["customer_idx", "product_idx"],
            "var": ["review_days"],
            "agg": ["mean", "sum", "median", "min", "max", "var", "std"],
        },
        {
            "key": ["customer_idx", "product_category"],
            "var": ["review_days", "ce_product_idx"],
            "agg": ["mean", "sum", "median", "min", "max", "var", "std"],
        },
    ]

    original_cols = train_test.columns
    groupby = GroupbyTransformer(param_dict=groupby_dict)
    train_test = groupby.transform(train_test)
    diff = DiffGroupbyTransformer(param_dict=groupby_dict)
    train_test = diff.transform(train_test)
    ratio = RatioGroupbyTransformer(param_dict=groupby_dict)
    train_test = ratio.transform(train_test)
    train_test[list(set(train_test.columns) - set(original_cols))].to_feather(
        "../input/feather/aggregation.ftr"
    )

    features = FeatureStore(
        feature_names=[
            "../input/feather/train_test.ftr",
            "../input/feather/count_encoding.ftr",
            "../input/feather/count_encoding_interact.ftr",
            "../input/feather/bow_tfidf.ftr",
            "../input/feather/aggregation.ftr",
        ],
        target_col=target_col[0],
    )

    X_train = features.X_train
    y_train = features.y_train
    X_test = features.X_test.iloc[:14597, :].reset_index(drop=True)
    X_final = features.X_test.iloc[14597:, :].reset_index(drop=True)
    print(X_train.shape, X_test.shape, X_final.shape)
    print(X_train.columns)

    fe_name = "fe014"
    Data.dump(X_train, f"../input/pickle/X_train_{fe_name}.pkl")
    Data.dump(y_train, f"../input/pickle/y_train_{fe_name}.pkl")
    Data.dump(X_test, f"../input/pickle/X_test_{fe_name}.pkl")
    Data.dump(X_final, f"../input/pickle/X_final_{fe_name}.pkl")
