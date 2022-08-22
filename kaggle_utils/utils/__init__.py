import json
import os
import time
from contextlib import contextmanager
from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger
from multiprocessing import Pool

import numpy as np
import pandas as pd
import requests
from IPython.display import Javascript, display
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted, column_or_1d

from .folds import *
from .load import *
from .sampling import *


def init_logger(file='log.txt'):
    dirs = '/'.join(file.split('/')[:-1])
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    LOGFORMAT = '%(asctime)s %(levelname)s %(message)s'
    logger = getLogger()
    fh_handler = FileHandler(file)
    fh_handler.setFormatter(Formatter(LOGFORMAT))
    logger.setLevel(INFO)
    logger.addHandler(fh_handler)
    return logger


@contextmanager
def timer(name, logger=None):
    if logger is not None:
        logger.info(f'[{name}] start')
    t0 = time.time()
    yield
    if logger is not None:
        logger.info(f'[{name}] done in {time.time() - t0:.0f} s')


def parallelize_dataframe(df, func, n_workers=4):
    df_split = np.array_split(df, n_workers)
    pool = Pool(n_workers)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def check_path(path):
    if os.path.exists(path):
        raise ValueError('{} already exists.'.format(path))


def save_notebook(current_ipynb_path, save_ipynb_path):
    print('Saving notebook...')
    display(Javascript('IPython.notebook.save_notebook()'), include=['application/javascript'])
    time.sleep(3)
    command = 'jupyter nbconvert --to notebook {} --output {}'.format(current_ipynb_path, save_ipynb_path)
    print('executing {}...'.format(command))
    os.system(command)
    print('Done.')


def submit(competition_name, file_path, comment='from API'):
    os.system(f'kaggle competitions submit -c {competition_name} -f {file_path} -m "{comment}"')
    time.sleep(60)
    tmp = os.popen(f'kaggle competitions submissions -c {COMPETITION_NAME} -v | head -n 2').read()
    col, values = tmp.strip().split('\n')
    message = 'SCORE!!!\n'
    for i, j in zip(col.split(','), values.split(',')):
        message += f'{i}: {j}\n'


class LINENotifyBot():
    API_URL = 'https://notify-api.line.me/api/notify'

    def __init__(self, access_token):
        self.__headers = {'Authorization': 'Bearer ' + access_token}

    def send(self, message, image=None, sticker_package_id=None, sticker_id=None):
        payload = {
            'message': message,
            'stickerPackageId': sticker_package_id,
            'stickerId': sticker_id,
        }
        files = {}
        if image is not None:
            files = {'imageFile': open(image, 'rb')}
        r = requests.post(
            LINENotifyBot.API_URL,
            headers=self.__headers,
            data=payload,
            files=files,
        )


class SpreadSheetBot():
    def __init__(self, endpoint):
        self.endpoint = endpoint

    def send(self, *args):
        requests.post(self.endpoint, json.dumps(args))


def change_dtype(dataframe, columns=None):
    if columns is None:
        columns = dataframe.columns

    for col in columns:
        col_type = dataframe[col].dtype

        if col_type != object:
            c_min = dataframe[col].min()
            c_max = dataframe[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    dataframe[col] = dataframe[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    dataframe[col] = dataframe[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    dataframe[col] = dataframe[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    dataframe[col] = dataframe[col].astype(np.int64)
            elif str(col_type)[:3] == 'flo':
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    # to avoid feather writting error
                    dataframe[col] = dataframe[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    dataframe[col] = dataframe[col].astype(np.float32)
                else:
                    dataframe[col] = dataframe[col].astype(np.float64)
    return dataframe


def to_category(train, cat=None):
    if cat is None:
        cat = [col for col in train.columns if train[col].dtype == 'object']
    for c in cat:
        train[c], uniques = pd.factorize(train[c])
    return train


class TolerantLabelEncoder(LabelEncoder):
    def __init__(self, ignore_unknown=True):
        """Tolerant label encoder which allows unseen labels"""
        self.ignore_unknown = ignore_unknown

    def fit(self, X, y=None):
        _, uniques = pd.factorize(X)
        self.classes_ = uniques
        self.classes_map_ = {k: v for k, v in zip(uniques, range(len(uniques)))}

    def transform(self, X, y=None):
        check_is_fitted(self, 'classes_')
        X = column_or_1d(X, warn=True)

        if not isinstance(X, pd.Series):
            X = pd.Series(X)
        else:
            X = pd.Series(X.values)

        indices = X.isin(self.classes_)
        nan_indices = X.isnull()

        if not self.ignore_unknown and not np.all(indices):
            raise ValueError('X contains new labels: %s'
                             % str(np.setdiff1d(X, self.classes_)))

        X_transformed = X.map(self.classes_map_)
        X_transformed[~indices] = len(self.classes_)
        X_transformed[nan_indices] = -1
        return X_transformed.values

    def fit_transform(self, X, y=None):
        codes, uniques = pd.factorize(X)
        self.classes_ = uniques
        self.classes_map_ = {k: v for k, v in zip(uniques, range(len(uniques)))}
        return codes

    def inverse_transform(self, X):
        check_is_fitted(self, 'classes_')
        labels = np.arange(len(self.classes_))
        indices = np.isin(X, labels)
        if not self.ignore_unknown and not np.all(indices):
            raise ValueError('X contains new labels: %s'
                             % str(np.setdiff1d(X, self.classes_)))

        X_transformed = np.asarray(self.classes_[X], dtype=object)
        X_transformed[~indices] = '_unknown'
        X_transformed[X == -1] = np.nan
        return X_transformed


class TolerantLabelEncoderOnMultipleCategories(LabelEncoder):
    def __init__(self, categorical_features, ignore_unknown=True):
        self.categorical_features = categorical_features
        self.ignore_unknown = ignore_unknown
        self.encoders = {}
        for f in self.categorical_features:
            self.encoders[f] = TolerantLabelEncoder(ignore_unknown)

    def fit(self, X):
        """
        Parameters
        ----------
        X : pd.DataFrame
        """
        for f in self.categorical_features:
            self.encoders[f].fit(X[f])
        return self

    def fit_transform(self, X):
        """
        Parameters
        ----------
        X : pd.DataFrame
        """
        for f in self.categorical_features:
            X[f] = self.encoders[f].fit_transform(X[f])
        return X

    def transform(self, X):
        """
        Parameters
        ----------
        X : pd.DataFrame
        """
        for f in self.categorical_features:
            X[f] = self.encoders[f].transform(X[f])
        return X

    def inverse_transform(self, X):
        """
        Parameters
        ----------
        X : pd.DataFrame
        """
        for f in self.categorical_features:
            X[f] = self.encoders[f].inverse_transform(X[f])
        return X
