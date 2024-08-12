#!/usr/bin/env python
# coding: utf-8

from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import os
import urllib.request
import zipfile
import sklearn.model_selection
import shutil


def download_movielens(PROJECT_ROOT="./", force=False):
    dataset_url = "https://files.grouplens.org/datasets/movielens/ml-10m.zip"
    raw_data_dir = os.path.join(PROJECT_ROOT, "data/raw")
    raw_dataset_dir = os.path.join(raw_data_dir, "ml-10m")
    raw_dataset_dir_temp = os.path.join(raw_data_dir, "ml-10M100K")
    raw_zipped_data_path = os.path.join(raw_data_dir, "ml-10m.zip")

    if force == False and os.path.exists(os.path.join(raw_dataset_dir, "ratings.dat")):
        print("Reuse existing ml-10m dataset")
        return

    if os.path.exists(raw_zipped_data_path):
        print(f"{raw_zipped_data_path} is being removed")
        shutil.rmtree(raw_zipped_data_path)
    if os.path.exists(raw_dataset_dir):
        print(f"{raw_dataset_dir} is being removed")
        shutil.rmtree(raw_dataset_dir)
    if os.path.exists(raw_dataset_dir_temp):
        print(f"{raw_dataset_dir_temp} is being removed")
        shutil.rmtree(raw_dataset_dir_temp)
        
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(raw_dataset_dir, exist_ok=True)
    # Download
    urllib.request.urlretrieve(dataset_url, raw_zipped_data_path)
    # Unpack
    zip_file = zipfile.ZipFile(raw_zipped_data_path)
    zip_file.extractall(raw_data_dir)
    os.rename(raw_dataset_dir_temp, raw_dataset_dir)


def prune_on_columns(df_train, df_test, columns=["uid", "iid"]):
    values_to_prune = {col: set(df_test[col].drop_duplicates()) - set(df_train[col].drop_duplicates()) for col in columns}
    while any(list(values_to_prune.values())):
        idx_to_prune = np.zeros(len(df_test), dtype=bool)
        for col, values in values_to_prune.items():
            idx_to_prune = idx_to_prune | df_test[col].isin(values)
        df_test = df_test.loc[~idx_to_prune]
        values_to_prune = {col: set(df_test[col].drop_duplicates()) - set(df_train[col].drop_duplicates()) for col in columns}
    return df_train, df_test


def split_train_test(df, test_ratio, seed, columns=["uid", "iid", "rating"]):
    rng = np.random.default_rng(seed)
    df_test = df.groupby(by=columns[0]).sample(frac=test_ratio, random_state=rng)
    df_train = df.loc[~df.index.isin(df_test.index)].copy()
    df_train, df_test = prune_on_columns(df_train, df_test)
    return df_train, df_test


def kfold_train_test_split_movielens(PROJECT_ROOT="./", num_folds=10, val_ratio=0.05, seed=12345):
    raw_data_dir = os.path.join(PROJECT_ROOT, "data/raw")
    raw_dataset_dir = os.path.join(raw_data_dir, "ml-10m")
    path_raw = os.path.join(raw_dataset_dir, "ratings.dat")
    df_raw = pd.read_csv(path_raw, sep="::", header=None)
    df_raw.columns = ["uid_orig", "iid_orig", "rating", "time"]
    df_raw["uid"]  = pd.CategoricalIndex(df_raw["uid_orig"], ordered=True).codes
    df_raw["iid"]  = pd.CategoricalIndex(df_raw["iid_orig"], ordered=True).codes
    li_cols = ["uid", "iid", "rating", "uid_orig", "iid_orig", "time"]
    df_raw = df_raw[li_cols].copy()
    skf = sklearn.model_selection.StratifiedKFold(num_folds, shuffle=True, random_state=seed)
    iter_folds_idx = skf.split(df_raw.index, df_raw["uid"])

    print("Start train_test_split (stratified with uid)", flush=True)
    for idx, (_idx_train, _idx_test) in tqdm(enumerate(iter_folds_idx), total=num_folds):
        out_fold_dir = os.path.join(PROJECT_ROOT, f"data/k-folds/ml-10m/{num_folds}-folds/{idx}")
        path_tr = os.path.join(out_fold_dir, "tr.csv")
        path_te = os.path.join(out_fold_dir, "te.csv")
        _df_train, _df_test = prune_on_columns(df_raw.loc[_idx_train], df_raw.loc[_idx_test])
        os.makedirs(out_fold_dir, exist_ok=True)
        _df_train.to_csv(path_tr, index=None)
        _df_test.to_csv(path_te, index=None)

        path_tr_tr = os.path.join(out_fold_dir, "tr_tr.csv")
        path_tr_te = os.path.join(out_fold_dir, "tr_te.csv")
        __df_train, __df_test = split_train_test(_df_train, val_ratio, seed)
        __df_train.to_csv(path_tr_tr, index=None)
        __df_test.to_csv(path_tr_te, index=None)



if __name__ == "__main__":
    PROJECT_ROOT = os.getenv("PROJECT_ROOT", "./")
    download_movielens(PROJECT_ROOT, force=False)
    kfold_train_test_split_movielens(PROJECT_ROOT, num_folds=10, seed=12345)

