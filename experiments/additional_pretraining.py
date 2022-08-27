import os
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.optim as optim
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

"""
Ref. Pytorch Lightningを使用したBERT文書分類モデルの実装 ( https://qiita.com/tchih11/items/7e97db29b95cf08fdda0 )
"""

class CreateDataset(Dataset):
    """
    DataFrameを下記のitemを保持するDatasetに変換。
    text(原文)、input_ids(tokenizeされた文章)、attention_mask、labels(ラベル)
    """

    def __init__(self, data, tokenizer, max_token_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        text = data_row[TEXT_COLUMN]
        labels = data_row[LABEL_COLUMN]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return dict(
            text=text,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.tensor(labels)
        )


class CreateDataModule(pl.LightningDataModule):
    """
    DataFrameからモデリング時に使用するDataModuleを作成
    """
    def __init__(self, train_df, valid_df, test_df, batch_size=16, max_token_len=512):
        super().__init__()
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')

    def setup(self, stage=None):
        self.train_dataset = CreateDataset(self.train_df, self.tokenizer, self.max_token_len)
        self.vaild_dataset = CreateDataset(self.valid_df, self.tokenizer, self.max_token_len)
        self.test_dataset = CreateDataset(self.test_df, self.tokenizer, self.max_token_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count())

    def val_dataloader(self):
        return DataLoader(self.vaild_dataset, batch_size=self.batch_size, num_workers=os.cpu_count())

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=os.cpu_count())


class TextClassifier(pl.LightningModule):
    def __init__(self, n_classes: int, n_epochs=None):
        super().__init__()

        # モデルの構造
        self.bert = AutoModel.from_pretrained('cl-tohoku/bert-base-japanese')
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)
        self.n_epochs = n_epochs
        self.criterion = nn.MSELoss()

        # BertLayerモジュールの最後を勾配計算ありに変更
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_grad = True

    # 順伝搬
    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        preds = self.classifier(output.pooler_output)
        loss = 0
        if labels is not None:
            loss = self.criterion(preds, labels)
        return loss, preds

    # trainのミニバッチに対して行う処理
    def training_step(self, batch, batch_idx):
        loss, preds = self.forward(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"],
                                    labels=batch["labels"])
        return {'loss': loss,
                'batch_preds': preds,
                'batch_labels': batch["labels"]}

    # validation、testでもtrain_stepと同じ処理を行う
    def validation_step(self, batch, batch_idx):
        loss, preds = self.forward(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"],
                                    labels=batch["labels"])
        return {'loss': loss,
                'batch_preds': preds,
                'batch_labels': batch["labels"]}

    def test_step(self, batch, batch_idx):
        loss, preds = self.forward(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"],
                                    labels=batch["labels"])
        return {'loss': loss,
                'batch_preds': preds,
                'batch_labels': batch["labels"]}

    # epoch終了時にvalidationのlossとaccuracyを記録
    def validation_epoch_end(self, outputs, mode="val"):
        # loss計算
        epoch_preds = torch.cat([x['batch_preds'] for x in outputs])
        epoch_labels = torch.cat([x['batch_labels'] for x in outputs])
        epoch_loss = self.criterion(epoch_preds, epoch_labels)
        self.log(f"{mode}_loss", epoch_loss, logger=True)

        # accuracy計算
        num_correct = (epoch_preds.argmax(dim=1) == epoch_labels).sum().item()
        epoch_accuracy = num_correct / len(epoch_labels)
        self.log(f"{mode}_accuracy", epoch_accuracy, logger=True)

    # testデータのlossとaccuracyを算出（validationの使いまわし）
    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, "test")

    # optimizerの設定
    def configure_optimizers(self):
        optimizer = optim.Adam([
            {'params': self.bert.encoder.layer[-1].parameters(), 'lr': 1e-5},
            {'params': self.classifier.parameters(), 'lr': 1e-5}
        ])

        return [optimizer]



if __name__ == "__main__":
    # # data_file2のみ感情極性（-2:強いネガティブ、-1:ネガティブ、0:ニュートラル、1:ポジティブ、2:強いポジティブ）が付与されたデータセット
    # data_file1 = "data/wrime/wrime-ver1.tsv"
    data_file2 = "data/wrime/wrime-ver2.tsv"
    checkpoint_dir = "data/wrime/checkpoints/"
    df = pd.read_csv(data_file2, sep='\t')

    pre_df = pd.DataFrame({'text': df['Sentence']})
    # pre_df['sentiment'] = df['Avg. Readers_Sentiment'].apply(convert_label)
    pre_df['sentiment'] = df['Avg. Readers_Sentiment']
    pre_df['sentiment'] = pre_df['sentiment'].astype('float32')
    train_df, valid_df, test_df = pre_df[:34000], pre_df[34000:], pre_df[34000:] # test_df利用しない。

    # 用意したDataFrameの文章、ラベルのカラム名
    TEXT_COLUMN = "text"
    LABEL_COLUMN = "sentiment"

    # 作ったDataFrameを渡してsetup
    data_module = CreateDataModule(train_df,valid_df,test_df)
    data_module.setup()


    # epoch数 (EarlyStopping前提)
    N_EPOCHS = 1000

    # モデルインスタンスを作成
    model = TextClassifier(n_classes=3,n_epochs=N_EPOCHS)

    # EarlyStoppingの設定
    # 3epochで'val_loss'が0.05以上減少しなければ学習をストップ
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.05,
        patience=3,
        mode='min')

    # モデルの保存先
    # epoch数に応じて、「epoch=0.ckpt」のような形で指定したディレクトリに保存される
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='{epoch}',
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    # Trainerに設定
    trainer = pl.Trainer(max_epochs=N_EPOCHS,
                         gpus=1,
                         callbacks=[checkpoint_callback, early_stop_callback])
    trainer.fit(model, data_module)
