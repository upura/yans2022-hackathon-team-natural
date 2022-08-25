import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score


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


if __name__ == "__main__":
    df = pd.read_json("../input/yans2022/training.jsonl", orient="records", lines=True)

    # ensembling oof score
    oof_train_lgbm = np.load("oof_train_lgbm.npy")
    oof_train_cat = np.load("oof_train_cat.npy")

    oof_train = oof_train_lgbm * 0.5 + oof_train_cat * 0.5
    df["pred_helpful_votes"] = oof_train

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

    y_pred_lgbm = np.load("y_pred_lgbm.npy")
    y_pred_cat = np.load("y_pred_cat.npy")
    y_test_final = y_pred_lgbm * 0.5 + y_pred_cat * 0.5
    df["pred_helpful_votes"] = y_test_final
    df_pred = convert_to_submit_format(df, "pred_helpful_votes", "pred")
    output_pred_file = "submission_run002.jsonl"
    df_pred.to_json(output_pred_file, orient="records", force_ascii=False, lines=True)
