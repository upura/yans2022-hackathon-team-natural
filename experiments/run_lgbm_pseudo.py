import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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


def run_lgbm(X_train, X_test, y_train, group_df, categorical_cols=[]):
    y_preds = []
    models = []
    oof_train = np.zeros((len(X_train),))
    cv = GroupKFold(n_splits=5)
    X_train = X_train.drop("product_idx", axis=1)
    X_test = X_test.drop("product_idx", axis=1)
    y_pred_lgbm = pd.Series(np.load("y_pred_lgbm.npy"))

    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 128,
        "max_depth": 8,
        "feature_fraction": 0.8,
        "subsample_freq": 1,
        "bagging_fraction": 0.7,
        "min_data_in_leaf": 10,
        "learning_rate": 0.01,
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
        X_tr = pd.concat([X_tr, X_test])
        y_tr = pd.concat([y_tr, y_pred_lgbm])

        lgb_train = lgb.Dataset(X_tr, y_tr, categorical_feature=categorical_cols)

        lgb_eval = lgb.Dataset(
            X_val, y_val, reference=lgb_train, categorical_feature=categorical_cols
        )

        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_eval],
            num_boost_round=1500,
            callbacks=[lgb.early_stopping(200), lgb.log_evaluation(200)],
        )

        oof_train[valid_index] = model.predict(
            X_val, num_iteration=model.best_iteration
        )
        joblib.dump(model, f"lgb_{fold_id}.pkl")
        models.append(model)

        y_pred = model.predict(X_test, num_iteration=model.best_iteration)
        y_preds.append(y_pred)

    return oof_train, y_preds, models


def visualize_importance(models, X_train):
    feature_importance_df = pd.DataFrame()
    for i, model in enumerate(models):
        _df = pd.DataFrame()
        _df["feature_importance"] = model.feature_importance()
        _df["column"] = X_train.columns
        _df["fold"] = i + 1
        feature_importance_df = pd.concat(
            [feature_importance_df, _df], axis=0, ignore_index=True
        )

    order = (
        feature_importance_df.groupby("column")
        .sum()[["feature_importance"]]
        .sort_values("feature_importance", ascending=False)
        .index[:50]
    )

    fig, ax = plt.subplots(figsize=(max(6, len(order) * 0.4), 7))
    sns.boxenplot(
        data=feature_importance_df,
        x="column",
        y="feature_importance",
        order=order,
        ax=ax,
        palette="viridis",
    )
    ax.tick_params(axis="x", rotation=90)
    ax.grid()
    fig.tight_layout()
    return fig, ax


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
    oof_train, y_preds, models = run_lgbm(
        X_train, X_test, y_train, X_train["product_idx"], categorical_cols
    )
    visualize_importance(models, X_train.drop("product_idx", axis=1))

    df = pd.read_json("../input/yans2022/training.jsonl", orient="records", lines=True)
    df["pred_helpful_votes"] = oof_train
    np.save("oof_train_lgbm", oof_train)

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
    np.save("y_pred_lgbm", np.average(y_preds, axis=0))
    df_pred = convert_to_submit_format(df, "pred_helpful_votes", "pred")
    output_pred_file = "submission_run000.jsonl"
    df_pred.to_json(output_pred_file, orient="records", force_ascii=False, lines=True)
