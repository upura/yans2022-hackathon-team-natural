# NLP若手の会 (YANS) 第17回シンポジウム ハッカソン (2022) 

- ECサイトAmazonのレビューデータを用いた特定の評価指標による[コンペティション](https://yans.anlp.jp/entry/yans2022hackathon)
- 最終評価で1位を獲得し、ハッカソンスポンサーのアマゾンウェブサービスジャパン合同会社から「Applied Scientist賞」も受賞

## Setup

Google Colab Pro+ を利用

```bash
# install MeCab
!apt-get -q -y install sudo file mecab libmecab-dev mecab-ipadic-utf8 git curl python-mecab > /dev/null
!git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git > /dev/null 
!echo yes | mecab-ipadic-neologd/bin/install-mecab-ipadic-neologd -n > /dev/null 2>&1
!pip install mecab-python3 > /dev/null

# check path to "ipadic-neologd" 
!echo `mecab-config --dicdir`"/mecab-ipadic-neologd"
cp /etc/mecabrc /usr/local/etc/

# Feature Engineering
!pip install -q catboost
!pip install -q category-encoders
!pip install -q node2vec==0.4.0
!pip install -q optuna
!pip install -q transformers

# LightGBM
!pip install -q lightgbm==3.3.2
```

## Run

```bash
cd path_to/experiments
!python additional_pretraining.py
!python create_bert_feature.py
!python create_features018.py
!python concat_features.py
!python run_lgbm_rank.py
```

- create_features018.py: 特徴量を生成
    - [kaggle_utils](https://github.com/Ynakatsuka/kaggle_utils) を一部で利用
- additional_pretraining.py
    - [WRIME: 主観と客観の感情分析データセット](https://github.com/ids-cv/wrime)のwrime-ver2.tsvを用いて学習するコード
    - Ref. [Pytorch Lightningを使用したBERT文書分類モデルの実装](https://qiita.com/tchih11/items/7e97db29b95cf08fdda0)
    - Early Stoppingを行った。（リーダーボードで利用したものはEpoch=2)
    - 損失関数はMSE Loss
- create_bert_feature.py
    - additional_pretraining.pyで作成したBERTモデルをロードして、baselineのpredictと同様にしてスコアを算出
    - baselineのコードのpredict.ipynbをもとに作成
- concat_feature.py: 利用する特徴量を結合
- run_lgbm_rank.py: LightGBM で学習・予測
