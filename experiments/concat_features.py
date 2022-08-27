import sys

sys.path.append("../")

import os
from typing import Any, List, Tuple

import joblib
import pandas as pd


def detect_delete_cols(
    train: pd.DataFrame, test: pd.DataFrame, escape_col: List[str], threshold: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Detect unnecessary columns for deleting
    Args:
        train (pd.DataFrame): train
        test (pd.DataFrame): test
        escape_col (List[str]): columns not encoded
        threshold (float): deleting threshold for correlations of columns
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train, test
    """
    unique_cols = list(train.columns[train.nunique() == 1])
    duplicated_cols = list(train.columns[train.T.duplicated()])

    buf = train.corr()
    counter = 0
    high_corr_cols = []
    try:
        for feat_a in [x for x in train.columns if x not in escape_col]:
            for feat_b in [x for x in train.columns if x not in escape_col]:
                if (
                    feat_a != feat_b
                    and feat_a not in high_corr_cols
                    and feat_b not in high_corr_cols
                ):
                    c = buf.loc[feat_a, feat_b]
                    if c > threshold:
                        counter += 1
                        high_corr_cols.append(feat_b)
                        print(
                            "{}: FEAT_A: {} FEAT_B: {} - Correlation: {}".format(
                                counter, feat_a, feat_b, c
                            )
                        )
    except:
        pass
    return unique_cols, duplicated_cols, high_corr_cols


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
    categorical_cols = [
        "product_category",
        "vine",
        "verified_purchase",
        "product_idx",
        "customer_idx",
    ]
    target_col = ["helpful_votes"]

    features = FeatureStore(
        feature_names=[
            "../input/feather/train_test.ftr",
            "../input/feather/count_encoding.ftr",
            "../input/feather/count_encoding_interact.ftr",
            "../input/feather/bow_tfidf.ftr",
            "../input/feather/aggregation.ftr",
            "../input/feather/bert_score.ftr",
        ],
        target_col=target_col[0],
    )

    X_train = features.X_train
    y_train = features.y_train
    X_test = features.X_test.iloc[:14597, :].reset_index(drop=True)
    X_final = features.X_test.iloc[14597:, :].reset_index(drop=True)
    print(X_train.shape, X_test.shape, X_final.shape)
    print(X_train.columns)

    unique_cols, duplicated_cols, high_corr_cols = detect_delete_cols(
        X_train, X_test, escape_col=categorical_cols, threshold=0.995
    )
    X_train = X_train.drop(unique_cols + duplicated_cols, axis=1)
    X_test = X_test.drop(unique_cols + duplicated_cols, axis=1)
    X_final = X_final.drop(unique_cols + duplicated_cols, axis=1)

    X_train = X_train.rename(columns={"pred_helpful_votes": "bert_sentiment"})
    X_test = X_test.rename(columns={"pred_helpful_votes": "bert_sentiment"})
    X_final = X_final.rename(columns={"pred_helpful_votes": "bert_sentiment"})
    X_train = X_train.drop("index", axis=1)
    X_test = X_test.drop("index", axis=1)
    X_final = X_final.drop("index", axis=1)

    fe_name = "fe018"
    Data.dump(X_train, f"../input/pickle/X_train_{fe_name}.pkl")
    Data.dump(y_train, f"../input/pickle/y_train_{fe_name}.pkl")
    Data.dump(X_test, f"../input/pickle/X_test_{fe_name}.pkl")
    Data.dump(X_final, f"../input/pickle/X_final_{fe_name}.pkl")
