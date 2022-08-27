import argparse
import os
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from collections import OrderedDict
from _my_lightning_modules import ReviewDataModule, ReviewRegressionNet

"""
- additional_pretraining.pyで作成したモデルを使って、特徴量を作る
- baselineのpredict.ipynbを編集したコード
"""

def remake_state_dict(state_dict):
    """
    - load_state_dictするために、BERTと最終fc層のパラメタを分ける
    - BERTの方は、後ほど net.pretrained_model.load_state_dict(state_dict) するために"bert."をkeyから削除する
    """
    new_state_dict = state_dict.copy()
    new_state_dict['state_dict'] = OrderedDict()
    for k, v in state_dict['state_dict'].items():
        # nk = k.replace('bert', 'pretrained_model')
        nk = k.replace('bert.', '')
        new_state_dict['state_dict'][nk] = v

    fc_weight = new_state_dict.pop('classifier.weight')
    fc_bias = new_state_dict.pop('classifier.bias')

    fc_state_dict = OrderedDict()
    fc_state_dict['weight'] = fc_weight
    fc_state_dict['bias'] = fc_bias

    return bert_state_dict, fc_state_dict



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ckpt_file", type=str)

    parser.add_argument(
        "--model_name", type=str, default="cl-tohoku/bert-base-japanese"
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=0)

    args_list = ["--input_file", "data/dataset_shared_initial/training.jsonl",
                "--output_dir", "data/emotion_bert/",
                "--ckpt_file", "data/emotion_bert/epoch=2.ckpt"]
    args = parser.parse_args(args_list)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    net = ReviewRegressionNet(args)

    state_dict = torch.load(args.ckpt_file, map_location='cpu')['state_dict']
    bert_state_dict, fc_state_dict = remake_state_dict(state_dict)
    net.pretrained_model.load_state_dict(bert_state_dict)
    net.fc.load_state_dict(fc_state_dict)

    dfs = []
    for filename in ['training.jsonl', 'leader_board.jsonl', 'final_result.jsonl']:
        args.input_file = os.path.join("data/dataset_shared_initial/", filename)

        dm = ReviewDataModule(args)
        trainer = pl.Trainer(gpus=[args.gpu], logger=False)
        pred = trainer.predict(net, dm)

        #スコアの作成
        df = pd.read_json(args.input_file, orient="records", lines=True)
        df.loc[:, "pred"] = sum([list(p.numpy().flatten()) for p in pred], [])
        df.loc[:, "pred_helpful_votes"] = df["pred"].apply(lambda x: np.exp(x) - 1)

        # output_file = args.output_dir + args.input_file.split("/")[-1]
        # df.to_json(output_file, orient="records", force_ascii=False, lines=True)

        df_bert_score = df[['pred_helpful_votes']].rename({'pred_helpful_votes': 'bert_score'})
        dfs.append(df_bert_score)


    bsdf = pd.concat(dfs)
    bsdf.reset_index(inplace=True)
    bsdf.to_feather("data/feather/bert_score.ftr")

