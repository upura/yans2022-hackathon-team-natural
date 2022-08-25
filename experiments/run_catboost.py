import catboost as cb
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from sklearn.model_selection import GroupKFold


def convert_to_submit_format(df, score_column, mode="pred"):
    output_list = []
    for product_idx in sorted(set(df["product_idx"])):
        df_product = df[df["product_idx"] == product_idx]
        scores = [
            {"review_idx": i, mode + "_score": s}
            for i, s in zip(df_product["review_idx"], df_product[score_column])
        ]
        output_list.append({"product_idx": product_idx, mode + "_list": scores})
    return pd.DataFrame(output_list)


def run_catboost(X_train, X_test, y_train, group_df, categorical_cols=[]):
    y_preds = []
    models = []
    oof_train = np.zeros((len(X_train),))
    cv = GroupKFold(n_splits=5)
    X_train = X_train.drop("product_idx", axis=1)
    X_test = X_test.drop("product_idx", axis=1)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 128,
        "max_depth": 8,
        "feature_fraction": 0.8,
        "subsample_freq": 1,
        "bagging_fraction": 0.7,
        "min_data_in_leaf": 10,
        "learning_rate": 0.05,
        "boosting": "gbdt",
        "lambda_l1": 0.4,
        "lambda_l2": 0.4,
        "verbosity": -1,
        "random_state": 42,
    }

    for fold_id, (train_index, valid_index) in enumerate(
        cv.split(X_train, groups=group_df)
    ):
        X_tr = X_train.loc[train_index, :]
        X_val = X_train.loc[valid_index, :]
        y_tr = y_train[train_index]
        y_val = y_train[valid_index]

        params = {
            "depth": 6,
            "learning_rate": 0.05,
            "iterations": 10000,
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "random_seed": 777,
            "allow_writing_files": False,
            "task_type": "CPU",
            "early_stopping_rounds": 200,
        }

        model = cb.CatBoostRegressor(**params)
        model.fit(
            X_tr,
            y_tr,
            cat_features=categorical_cols,
            eval_set=(X_val, y_val),
            verbose=200,
            use_best_model=True,
            plot=False,
        )

        oof_train[valid_index] = model.predict(X_val)
        joblib.dump(model, f"lgb_{fold_id}.pkl")
        models.append(model)

        y_pred = model.predict(X_test)
        y_preds.append(y_pred)

    return oof_train, y_preds, models


if __name__ == "__main__":
    X_train = joblib.load("../input/pickle/X_train_fe004.pkl")
    y_train = joblib.load("../input/pickle/y_train_fe004.pkl")
    X_test = joblib.load("../input/pickle/X_test_fe004.pkl")
    X_test.head()

    categorical_cols = [
        "product_category",
        "star_rating",
        "vine",
        "verified_purchase",
        "customer_idx",
    ]
    oof_train, y_preds, models = run_catboost(
        X_train, X_test, y_train, X_train["product_idx"], categorical_cols
    )

    df = pd.read_json("../input/yans2022/training.jsonl", orient="records", lines=True)
    df["pred_helpful_votes"] = oof_train
    np.save("oof_train_cat", oof_train)

    df_pred = convert_to_submit_format(df, "pred_helpful_votes", "pred")

    df_true = convert_to_submit_format(df, "helpful_votes", "true")
    df_merge = pd.merge(df_pred, df_true, on="product_idx")

    sum_ndcg = 0
    for df_dict in df_merge.to_dict("records"):
        df_eval = pd.merge(
            pd.DataFrame(df_dict["pred_list"]),
            pd.DataFrame(df_dict["true_list"]),
            on="review_idx",
        )
        try:
            ndcg = ndcg_score([df_eval["true_score"]], [df_eval["pred_score"]], k=5)
        except ValueError:
            ndcg = 0
        sum_ndcg += ndcg
    print(sum_ndcg / len(df_merge))

    df = pd.read_json(
        "../input/yans2022/leader_board.jsonl", orient="records", lines=True
    )
    df["pred_helpful_votes"] = np.average(y_preds, axis=0)
    np.save("y_pred_cat", np.average(y_preds, axis=0))
    df_pred = convert_to_submit_format(df, "pred_helpful_votes", "pred")
    output_pred_file = "submission_run000.jsonl"
    df_pred.to_json(output_pred_file, orient="records", force_ascii=False, lines=True)
