# NLP若手の会 (YANS) 第17回シンポジウム ハッカソン (2022) 

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
!python create_features014.py
!python run_lgbm_rank.py
```

- create_features014.py: 特徴量を生成
- run_lgbm_rank.py: LightGBM で学習・予測
- additional\_pretraining.py
    - [WRIME: 主観と客観の感情分析データセット](https://github.com/ids-cv/wrime)のwrime-ver2.tsvを用いて学習するコード
    - Ref. [Pytorch Lightningを使用したBERT文書分類モデルの実装](https://qiita.com/tchih11/items/7e97db29b95cf08fdda0)
    - Early Stoppingを行った。（リーダーボードで利用したものはEpoch=2)
    - 損失関数はMSE Loss
- create\_bert\_feature.py
    - additional\_pretraining.pyで作成したBERTモデルをロードして、baselineのpredictと同様にしてスコアを算出
    - baselineのコードのpredict.ipynbをもとに作成
