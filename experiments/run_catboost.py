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


def run_catboost(X_train, X_test, X_final, y_train, group_df, categorical_cols=[]):
    y_preds = []
    y_finals = []
    models = []
    oof_train = []
    cv = GroupKFold(n_splits=5)
    X_test = X_test.drop("product_idx", axis=1)
    X_final = X_final.drop("product_idx", axis=1)

    for fold_id, (train_index, valid_index) in enumerate(
        cv.split(X_train, groups=group_df)
    ):
        X_tr = X_train.loc[train_index, :]
        X_val = X_train.loc[valid_index, :]
        y_tr = y_train[train_index]
        y_val = y_train[valid_index]

        X_tr["target"] = y_tr
        X_val["target"] = y_val

        X_tr = X_tr.sort_values("product_idx").reset_index(drop=True)
        X_val = X_val.sort_values("product_idx").reset_index(drop=True)
        y_tr = X_tr["target"]
        y_val = X_val["target"]

        train_pool = cb.Pool(
            data=X_tr.drop(["target", "product_idx"], axis=1),
            label=y_tr,
            group_id=X_tr["product_idx"],
            cat_features=categorical_cols,
        )

        eval_pool = cb.Pool(
            data=X_val.drop(["target", "product_idx"], axis=1),
            label=y_val,
            group_id=X_val["product_idx"],
            cat_features=categorical_cols,
        )

        params = {
            "depth": 8,
            "learning_rate": 0.05,
            "iterations": 10000,
            "loss_function": "YetiRank",
            "eval_metric": "NDCG:top=5",
            "random_seed": 777,
            "allow_writing_files": False,
            "task_type": "CPU",
            "early_stopping_rounds": 200,
        }
        model = cb.CatBoost(params)
        model.fit(
            train_pool,
            eval_set=eval_pool,
            verbose=200,
            use_best_model=True,
            plot=False,
        )

        X_val["pred_helpful_votes"] = model.predict(
            X_val.drop(["target", "product_idx"], axis=1),
        )
        oof_df = X_val[["product_idx", "target", "pred_helpful_votes"]]
        oof_train.append(oof_df)

        joblib.dump(model, f"lgb_{fold_id}.pkl")
        models.append(model)

        y_pred = model.predict(X_test)
        y_preds.append(y_pred)
        y_final = model.predict(X_final)
        y_finals.append(y_final)

    return oof_train, y_preds, y_finals, models


if __name__ == "__main__":
    run_name = "cat000"
    X_train = joblib.load("../input/pickle/X_train_fe014.pkl")
    y_train = joblib.load("../input/pickle/y_train_fe014.pkl")
    X_test = joblib.load("../input/pickle/X_test_fe014.pkl")
    X_final = joblib.load("../input/pickle/X_final_fe014.pkl")
    y_train = y_train.astype(int)
    print(X_train.shape, X_test.shape, X_final.shape)

    categorical_cols = [
        "product_category",
        "vine",
        "verified_purchase",
        "customer_idx",
    ]
    oof_train, y_preds, y_finals, models = run_catboost(
        X_train, X_test, X_final, y_train, X_train["product_idx"], categorical_cols
    )

    df = pd.concat(oof_train)
    df["review_idx"] = np.arange(len(df))
    df.to_csv(f"oof_df{run_name}.csv", index=False)
    df_pred = convert_to_submit_format(df, "pred_helpful_votes", "pred")
    df_true = convert_to_submit_format(df, "target", "true")
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
    np.save(f"y_pred{run_name}", np.average(y_preds, axis=0))
    df_pred = convert_to_submit_format(df, "pred_helpful_votes", "pred")
    output_pred_file = f"submission_lb{run_name}.jsonl"
    df_pred.to_json(output_pred_file, orient="records", force_ascii=False, lines=True)

    df = pd.read_json(
        "../input/yans2022/final_result.jsonl", orient="records", lines=True
    )
    df["pred_helpful_votes"] = np.average(y_finals, axis=0)
    np.save(f"y_final{run_name}", np.average(y_finals, axis=0))
    df_pred = convert_to_submit_format(df, "pred_helpful_votes", "pred")
    output_pred_file = f"submit_final_result{run_name}.jsonl"
    df_pred.to_json(output_pred_file, orient="records", force_ascii=False, lines=True)
