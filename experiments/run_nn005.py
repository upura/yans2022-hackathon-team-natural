import argparse
import json
import os

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import ndcg_score
from sklearn.model_selection import GroupKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer

FOLD_ID = 0
EXPERIMENT_NAME = "predict_helpful_votes"
MODEL_PATH = "microsoft/infoxlm-large"
RUN_NAME = f"{MODEL_PATH}_lr5e-6"


class ReviewDataset(Dataset):
    def __init__(self, df, tokenizer):

        self.out_inputs = []
        for df_dict in df.to_dict("records"):
            inputs = tokenizer(df_dict["review_body"], truncation=True, max_length=512)
            if "label" in df_dict:
                inputs.update({"label": df_dict["label"]})
            inputs.update({"product_idx": df_dict["product_idx"]})
            inputs.update({"review_idx": df_dict["review_idx"]})
            inputs.update({"customer_idx": df_dict["review_idx"]})
            self.out_inputs.append(inputs)

    def __len__(self):
        return len(self.out_inputs)

    def __getitem__(self, idx):
        return self.out_inputs[idx]


class ReviewDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.df = pd.read_json(hparams.input_file, orient="records", lines=True)
        self.df["review_body"] = (
            self.df["customer_idx"].astype(str) + "[SEP]" + self.df["review_body"]
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name)
        self.batch_size = hparams.batch_size

    def _make_label(self):
        self.df.loc[:, "label"] = self.df.loc[:, "helpful_votes"]

    def setup(self, stage):
        if stage == "fit":
            self._make_label()
            df_train = self.df[self.df["sets"] == "training-train"]
            df_val = self.df[self.df["sets"] == "training-val"]

            self.train_dataset = ReviewDataset(df_train, self.tokenizer)
            self.val_dataset = ReviewDataset(df_val, self.tokenizer)
        elif stage == "test":
            self.test_dataset = ReviewDataset(self.df, self.tokenizer)
        elif stage == "predict":
            self.predict_dataset = ReviewDataset(self.df, self.tokenizer)

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, True)

    def val_dataloader(self):
        return self._get_dataloader(self.val_dataset, False)

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, False)

    def predict_dataloader(self):
        return self._get_dataloader(self.predict_dataset, False)

    def _get_dataloader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            shuffle=shuffle,
            num_workers=8,
        )

    def _collate_fn(self, batch):
        output_dict = {}
        for i in ["input_ids", "attention_mask"]:
            output_dict[i] = nn.utils.rnn.pad_sequence(
                [torch.LongTensor(b[i]) for b in batch],
                batch_first=True,
            )
        output_dict["product_idx"] = torch.IntTensor(
            [[b["product_idx"]] for b in batch]
        )
        output_dict["review_idx"] = torch.IntTensor([[b["review_idx"]] for b in batch])
        if "label" in batch[0]:
            output_dict["label"] = torch.FloatTensor([[b["label"]] for b in batch])
        return output_dict


class ReviewRegressionNet(pl.LightningModule):
    def __init__(self, hparams):
        super(ReviewRegressionNet, self).__init__()
        self.save_hyperparameters(hparams)
        model_config = AutoConfig.from_pretrained(self.hparams.model_name)
        model_config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": 0.0,
                "layer_norm_eps": 1e-7,
            }
        )
        self.pretrained_model = AutoModel.from_pretrained(
            self.hparams.model_name, config=model_config
        )
        self.fc = nn.Linear(self.pretrained_model.config.hidden_size * 4, 1)
        self.criterion = nn.MSELoss()

    def forward(self, batch):
        output = self.pretrained_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        sequence_concat = torch.cat(
            [output["hidden_states"][-1 * i][:, 0] for i in range(1, 4 + 1)],
            dim=1,
        )
        return self.fc(sequence_concat)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams.learning_rate)

    def _compute_rmse(self, output, true):
        return torch.sqrt(self.criterion(output, true))

    def training_step(self, batch, _):
        output = self.forward(batch)
        loss = self.criterion(output, batch["label"])
        rmse = self._compute_rmse(output, batch["label"])
        return {
            "loss": loss,
            "rmse": rmse,
            "y_pred": output,
            "label": batch["label"],
            "product_idx": batch["product_idx"],
            "review_idx": batch["review_idx"],
        }

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_rmse = torch.stack([x["rmse"] for x in outputs]).mean()
        y_preds = torch.cat([x["y_pred"] for x in outputs])
        labels = torch.cat([x["label"] for x in outputs])
        product_idxs = torch.cat([x["product_idx"] for x in outputs])
        review_idxs = torch.cat([x["review_idx"] for x in outputs])
        self.log("train_loss", avg_loss.detach(), on_epoch=True, prog_bar=True)
        self.log("train_rmse", avg_rmse.detach(), on_epoch=True, prog_bar=True)

        df = pd.DataFrame(
            {
                "review_idx": list(review_idxs.detach().cpu().numpy().reshape(-1)),
                "product_idx": list(product_idxs.detach().cpu().numpy().reshape(-1)),
                "helpful_votes": list(labels.detach().cpu().numpy().reshape(-1)),
            }
        )
        df["pred_helpful_votes"] = y_preds.detach().cpu().numpy().reshape(-1)
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
        ndcg = sum_ndcg / len(df_merge)
        self.log("train_ndcg", ndcg, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, _):
        output = self.forward(batch)
        loss = self.criterion(output, batch["label"])
        rmse = self._compute_rmse(output, batch["label"])
        return {
            "loss": loss,
            "rmse": rmse,
            "y_pred": output,
            "label": batch["label"],
            "product_idx": batch["product_idx"],
            "review_idx": batch["review_idx"],
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_rmse = torch.stack([x["rmse"] for x in outputs]).mean()
        y_preds = torch.cat([x["y_pred"] for x in outputs])
        labels = torch.cat([x["label"] for x in outputs])
        product_idxs = torch.cat([x["product_idx"] for x in outputs])
        review_idxs = torch.cat([x["review_idx"] for x in outputs])
        self.log("val_loss", avg_loss.detach(), on_epoch=True, prog_bar=True)
        self.log("val_rmse", avg_rmse.detach(), on_epoch=True, prog_bar=True)

        df = pd.DataFrame(
            {
                "review_idx": list(review_idxs.detach().cpu().numpy().reshape(-1)),
                "product_idx": list(product_idxs.detach().cpu().numpy().reshape(-1)),
                "helpful_votes": list(labels.detach().cpu().numpy().reshape(-1)),
            }
        )
        df["pred_helpful_votes"] = y_preds.detach().cpu().numpy().reshape(-1)
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
        val_ndcg = sum_ndcg / len(df_merge)
        self.log("val_ndcg", val_ndcg, on_epoch=True, prog_bar=True)
        return {"val_loss": avg_loss, "val_rmse": avg_rmse, "val_ndcg": val_ndcg}

    def test_step(self, batch, _):
        output = self.forward(batch)
        loss = self.criterion(output, batch["label"])
        rmse = self._compute_rmse(output, batch["label"])
        return {"loss": loss, "rmse": rmse}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_rmse = torch.stack([x["rmse"] for x in outputs]).mean()
        self.log("test_loss", avg_loss.detach(), on_epoch=True, prog_bar=True)
        self.log("test_rmse", avg_rmse.detach(), on_epoch=True, prog_bar=True)
        return {"test_loss": avg_loss, "test_rmse": avg_rmse}

    def predict_step(self, batch, _):
        return self.forward(batch)


def preprocessing(mode: str = ""):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_train", type=int)
    parser.add_argument("--n_val", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=0)

    args_list = [
        "--input_file",
        "../input/yans2022/training.jsonl",
        "--output_dir",
        "./data/preprocessing_shared/",
        "--n_train",
        "10",
        "--n_val",
        "10",
    ]
    args = parser.parse_args(args_list)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    df = pd.read_json(args.input_file, orient="records", lines=True)
    if mode == "debug":
        df = df[:1000]

    group_kfold = GroupKFold(n_splits=11)
    for fold_id, (tr_idx, dev_idx) in enumerate(
        group_kfold.split(df, df["helpful_votes"], df["product_idx"])
    ):
        if fold_id == FOLD_ID:
            df.loc[tr_idx, "sets"] = "training-train"
            df.loc[dev_idx, "sets"] = "training-val"
    df_sets = df.copy()
    print(df_sets["sets"].value_counts())

    df_sets.to_json(
        args.output_dir + "training.jsonl",
        orient="records",
        force_ascii=False,
        lines=True,
    )

    df_tr = df_sets[df_sets["sets"].str.contains("-train")]
    df_tr.to_json(
        args.output_dir + "training-train.jsonl",
        orient="records",
        force_ascii=False,
        lines=True,
    )

    df_val = df_sets[df_sets["sets"].str.contains("-val")]
    df_val.to_json(
        args.output_dir + "training-val.jsonl",
        orient="records",
        force_ascii=False,
        lines=True,
    )


def training():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_model_dir", type=str, required=True)
    parser.add_argument("--output_csv_dir", type=str, required=True)
    parser.add_argument("--output_mlruns_dir", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, default="predict_helpful_votes")
    parser.add_argument("--run_name", type=str, default=f"{MODEL_PATH}_lr1e-5")
    parser.add_argument("--model_name", type=str, default=MODEL_PATH)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])

    args_list = [
        "--input_file",
        "./data/preprocessing_shared/training.jsonl",
        "--output_model_dir",
        "./data/train/model/",
        "--output_csv_dir",
        "./data/train/csv/",
        "--output_mlruns_dir",
        "./data/train/mlruns/",
        "--max_epochs",
        "3",
        "--learning_rate",
        "5e-6",
        "--gpus",
        "0",
    ]
    args = parser.parse_args(args_list)

    if not os.path.isdir(args.output_model_dir):
        os.makedirs(args.output_model_dir)
    if not os.path.isdir(args.output_csv_dir):
        os.makedirs(args.output_csv_dir)
    if not os.path.isdir(args.output_mlruns_dir):
        os.makedirs(args.output_mlruns_dir)

    dm = ReviewDataModule(args)
    net = ReviewRegressionNet(args)
    output_model_dir = args.output_model_dir + args.experiment_name + "/"

    trainer = pl.Trainer(
        precision=16,
        accelerator="gpu",
        devices=args.gpus,
        max_epochs=args.max_epochs,
        val_check_interval=10000,
        accumulate_grad_batches=4,
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="val_ndcg", patience=3, mode="max"),
            pl.callbacks.ModelCheckpoint(
                dirpath=output_model_dir,
                filename=args.run_name,
                verbose=True,
                monitor="val_ndcg",
                mode="max",
                save_top_k=1,
            ),
        ],
        logger=[
            pl.loggers.csv_logs.CSVLogger(
                save_dir=args.output_csv_dir,
                name=args.experiment_name,
                version=args.run_name,
            ),
        ],
    )
    trainer.fit(net, dm)


def predicting(mode: str):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ckpt_file", type=str)

    parser.add_argument("--model_name", type=str, default=MODEL_PATH)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gpu", type=int, default=0)

    if mode == "val":
        args_list = [
            "--input_file",
            "./data/preprocessing_shared/training-val.jsonl",
            "--output_dir",
            "./data/predict/" + EXPERIMENT_NAME + "/" + RUN_NAME + "/",
            "--ckpt_file",
            "./data/train/model/" + EXPERIMENT_NAME + "/" + RUN_NAME + ".ckpt",
        ]
    elif mode == "lb":
        args_list = [
            "--input_file",
            "../input/yans2022/leader_board.jsonl",
            "--output_dir",
            "./data/predict/" + EXPERIMENT_NAME + "/" + RUN_NAME + "/",
            "--ckpt_file",
            "./data/train/model/" + EXPERIMENT_NAME + "/" + RUN_NAME + ".ckpt",
        ]

    args = parser.parse_args(args_list)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    dm = ReviewDataModule(args)
    net = ReviewRegressionNet(args)
    trainer = pl.Trainer(gpus=[args.gpu], logger=False)
    if args.ckpt_file is None:
        pred = trainer.predict(net, dm)
    else:
        pred = trainer.predict(net, dm, ckpt_path=args.ckpt_file)

    df = pd.read_json(args.input_file, orient="records", lines=True)
    df.loc[:, "pred"] = sum([list(p.numpy().flatten()) for p in pred], [])
    df.loc[:, "pred_helpful_votes"] = df["pred"]

    output_file = args.output_dir + args.input_file.split("/")[-1]
    df.to_json(output_file, orient="records", force_ascii=False, lines=True)


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


def evaluating(mode: str):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    if mode == "val":
        args_list = [
            "--input_file",
            "./data/predict/"
            + EXPERIMENT_NAME
            + "/"
            + RUN_NAME
            + "/training-val.jsonl",
            "--output_dir",
            "./data/evaluation/" + EXPERIMENT_NAME + "/" + RUN_NAME + "/",
        ]
    elif mode == "lb":
        args_list = [
            "--input_file",
            "./data/predict/"
            + EXPERIMENT_NAME
            + "/"
            + RUN_NAME
            + "/leader_board.jsonl",
            "--output_dir",
            "./data/evaluation/" + EXPERIMENT_NAME + "/" + RUN_NAME + "/",
        ]

    args = parser.parse_args(args_list)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    df = pd.read_json(args.input_file, orient="records", lines=True)

    df_pred = convert_to_submit_format(df, "pred_helpful_votes", "pred")
    output_pred_file = args.output_dir + "submit_" + args.input_file.split("/")[-1]
    df_pred.to_json(output_pred_file, orient="records", force_ascii=False, lines=True)

    if "helpful_votes" in df.columns:
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

        output_eval_file = (
            args.output_dir
            + "eval_"
            + args.input_file.split("/")[-1].replace(".jsonl", ".json")
        )
        with open(output_eval_file, "w") as f:
            json.dump(
                {"ndcg@5": sum_ndcg / len(df_merge)}, f, indent=4, ensure_ascii=False
            )
        print(sum_ndcg / len(df_merge))


if __name__ == "__main__":
    preprocessing(mode="")
    training()
    # predicting(mode="val")
    # evaluating(mode="val")
    # predicting(mode="lb")
    # evaluating(mode="lb")
