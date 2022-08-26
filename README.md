# NLP若手の会 (YANS) 第17回シンポジウム ハッカソン (2022) 

## Setup

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
!pip install lightgbm==3.3.2
```

## Run

```bash
cd path_to/experiments
!python create_features.py
!python run_lgbm_rank.py
```

- create_features.py: 特徴量を生成
- run_lgbm_rank.py: LightGBM で学習・予測
