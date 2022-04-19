import cudf
from cuml import ForestInference

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import time
import gc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import datetime
import itertools
import os
from contextlib import redirect_stdout
from tqdm.auto import tqdm


def apk(actual, predicted, k=10):
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


path = "../input/h-and-m-personalized-fashion-recommendations/"
exp_name = os.path.basename(__file__).split(".")[0]
save_path = f"./{exp_name}"
os.makedirs(save_path, exist_ok=True)

rand = 64
lgb_params = {
    "objective": "binary",
    "boosting": "gbdt",
    "max_depth": -1,
    "num_leaves": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "bagging_seed": rand,
    "learning_rate": 0.05,
    "feature_fraction": 0.6,
    #     "feature_seed": rand,
    "min_data_in_leaf": 100,
    "lambda_l1": 0,
    "lambda_l2": 0,
    "random_state": rand,
    "metric": "average_precision",  # "auc",  # "binary_logloss",
    "verbose": -1,
    "n_jobs": 18,
}

lgb_train_params = {"early_stopping_rounds": 100, "verbose_eval": 100}

tran_dtypes = {
    "t_dat": "str",
    "customer_id": "str",
    "article_id": "int",
    "product_code": "int",
    "price": "float",
    "sales_channel_id": "int",
}
art_dtypes = {
    "article_id": "int",
    "product_code": "int",
    "product_type_no": "int",
    "graphical_appearance_no": "int",
    "colour_group_code": "int",
    "department_no": "int",
    "index_code": "str",
    "index_group_no": "int",
    "section_no": "int",
    "garment_group_no": "int",
}
cust_dtypes = {"customer_id": "str"}

N = 6000
N_div = 100
n_iter = 1
tmp_top = 200
len_hist = 366
n_round = 2000
n_splits = 1
n_week = 10
tr_set = [(i * 7) + 1 for i in range(n_week)]
print(tr_set)
len_tr = 7
nobuy = 20
mode = "cv"
iwata_version = "v1"
cache_dir = f"./cache/{iwata_version}"
os.makedirs(cache_dir, exist_ok=True)


def read_data(day_oldest):
    df_art = pd.read_csv(path + "articles.csv", dtype=art_dtypes)
    le = LabelEncoder()
    le.fit(df_art["index_code"].unique())
    df_art["index_code"] = le.transform(df_art["index_code"])

    df_trans = pd.read_csv(path + "transactions_train.csv", dtype=tran_dtypes)
    df_trans["t_dat"] = pd.to_datetime(df_trans["t_dat"], format="%Y-%m-%d")
    df_trans = df_trans.query(f"t_dat >= '{day_oldest}'").copy()
    df_trans = df_trans.drop_duplicates(["customer_id", "article_id", "t_dat"])
    df_trans = df_trans.merge(
        df_art[
            [
                "article_id",
                "product_code",
                "product_type_no",
                "graphical_appearance_no",
                "colour_group_code",
                "department_no",
                "index_code",
                "index_group_no",
                "section_no",
                "garment_group_no",
            ]
        ],
        how="left",
        on="article_id",
    )

    df_cust = pd.read_csv(path + "customers.csv", dtype=cust_dtypes)
    df_cust["age"] = df_cust["age"].fillna(df_cust["age"].mean())
    df_cust[["FN", "Active"]] = df_cust[["FN", "Active"]].fillna(0)
    df_cust["club_member_status"] = df_cust["club_member_status"].apply(
        lambda x: 1 if x == "ACTIVE" else 0
    )
    df_cust["fashion_news_frequency"] = df_cust["fashion_news_frequency"].apply(
        lambda x: 0 if x == "NONE" else 1
    )

    return df_trans, df_art, df_cust


def feat_store(df_trans, l_cust, ds, de, dsr, der, dsh, deh):
    feat = {}

    df_trans_yesterday = df_trans.query("(t_dat == @der)")
    df_trans_recent = df_trans.query("(t_dat >= @dsr) and (t_dat <= @der)")
    df_trans_hist = df_trans.query("(t_dat >= @dsh) and (t_dat <= @deh)")

    feat["art_buy_hist"] = df_trans_hist.groupby(["article_id"])["t_dat"].agg(
        art_buy_hist="count"
    )
    feat["art_buy_recent"] = df_trans_recent.groupby(["article_id"])["t_dat"].agg(
        art_buy_recent="count"
    )
    feat["art_buy_yesterday"] = df_trans_yesterday.groupby(["article_id"])["t_dat"].agg(
        art_buy_yesterday="count"
    )
    df_buy1 = (
        df_trans_hist.groupby("article_id")["customer_id"]
        .nunique()
        .reset_index()
        .rename(columns={"customer_id": "cnt_buy1"})
    )
    df_buy2 = df_trans_hist[
        df_trans_hist.duplicated(["customer_id", "article_id"])
    ].copy()
    df_buy2 = df_buy2.drop_duplicates(["customer_id", "article_id"])
    df_buy2 = (
        df_buy2.groupby("article_id")["article_id"].agg(cnt_buy2="count").reset_index()
    )
    df_buy = pd.merge(df_buy1, df_buy2, how="left", on="article_id").fillna(0)
    df_buy["rebuy_rate"] = df_buy["cnt_buy2"] / df_buy["cnt_buy1"]
    feat["rebuy_rate"] = df_buy[["article_id", "rebuy_rate"]]

    df_trans_yesterday = df_trans_yesterday.query("(customer_id in @l_cust)")
    df_trans_recent = df_trans_recent.query("(customer_id in @l_cust)")
    df_trans_hist = df_trans_hist.query("(customer_id in @l_cust)")
    feat["rate_sales_channel_hist"] = df_trans_hist.groupby(["customer_id"])[
        "sales_channel_id"
    ].agg(rate_sales_channel_hist="mean")
    feat["rate_sales_channel_recent"] = df_trans_recent.groupby(["customer_id"])[
        "sales_channel_id"
    ].agg(rate_sales_channel_recent="mean")
    feat["n_buy_hist"] = df_trans_hist.groupby(["customer_id", "article_id"])[
        "t_dat"
    ].agg(n_buy_hist="count")
    feat["n_buy_recent"] = df_trans_recent.groupby(["customer_id", "article_id"])[
        "t_dat"
    ].agg(n_buy_recent="count")
    feat["days_after_buy"] = df_trans_hist.groupby(["customer_id", "article_id"])[
        "t_dat"
    ].agg(days_after_buy=lambda x: (ds - max(x)).days)
    feat["n_buy_hist_all"] = df_trans_hist.groupby(["customer_id"])["t_dat"].agg(
        n_buy_hist_all="count"
    )
    feat["n_buy_recent_all"] = df_trans_recent.groupby(["customer_id"])["t_dat"].agg(
        n_buy_recent_all="count"
    )
    feat["days_after_buy_all"] = df_trans_hist.groupby(["customer_id"])["t_dat"].agg(
        days_after_buy_all=lambda x: (ds - max(x)).days
    )
    feat["n_buy_hist_prod"] = df_trans_hist.groupby(["customer_id", "product_code"])[
        "t_dat"
    ].agg(n_buy_hist_prod="count")
    feat["n_buy_recent_prod"] = df_trans_recent.groupby(
        ["customer_id", "product_code"]
    )["t_dat"].agg(n_buy_recent_prod="count")
    feat["days_after_buy_prod"] = df_trans_hist.groupby(
        ["customer_id", "product_code"]
    )["t_dat"].agg(days_after_buy_prod=lambda x: (ds - max(x)).days)
    feat["n_buy_hist_ptype"] = df_trans_hist.groupby(
        ["customer_id", "product_type_no"]
    )["t_dat"].agg(n_buy_hist_ptype="count")
    feat["n_buy_recent_ptype"] = df_trans_recent.groupby(
        ["customer_id", "product_type_no"]
    )["t_dat"].agg(n_buy_recent_ptype="count")
    feat["days_after_buy_ptype"] = df_trans_hist.groupby(
        ["customer_id", "product_type_no"]
    )["t_dat"].agg(days_after_buy_ptype=lambda x: (ds - max(x)).days)
    feat["n_buy_hist_graph"] = df_trans_hist.groupby(
        ["customer_id", "graphical_appearance_no"]
    )["t_dat"].agg(n_buy_hist_graph="count")
    feat["n_buy_recent_graph"] = df_trans_recent.groupby(
        ["customer_id", "graphical_appearance_no"]
    )["t_dat"].agg(n_buy_recent_graph="count")
    feat["days_after_buy_graph"] = df_trans_hist.groupby(
        ["customer_id", "graphical_appearance_no"]
    )["t_dat"].agg(days_after_buy_graph=lambda x: (ds - max(x)).days)
    feat["n_buy_hist_col"] = df_trans_hist.groupby(
        ["customer_id", "colour_group_code"]
    )["t_dat"].agg(n_buy_hist_col="count")
    feat["n_buy_recent_col"] = df_trans_recent.groupby(
        ["customer_id", "colour_group_code"]
    )["t_dat"].agg(n_buy_recent_col="count")
    feat["days_after_buy_col"] = df_trans_hist.groupby(
        ["customer_id", "colour_group_code"]
    )["t_dat"].agg(days_after_buy_col=lambda x: (ds - max(x)).days)
    feat["n_buy_hist_dep"] = df_trans_hist.groupby(["customer_id", "department_no"])[
        "t_dat"
    ].agg(n_buy_hist_dep="count")
    feat["n_buy_recent_dep"] = df_trans_recent.groupby(
        ["customer_id", "department_no"]
    )["t_dat"].agg(n_buy_recent_dep="count")
    feat["days_after_buy_dep"] = df_trans_hist.groupby(
        ["customer_id", "department_no"]
    )["t_dat"].agg(days_after_buy_dep=lambda x: (ds - max(x)).days)
    feat["n_buy_hist_idx"] = df_trans_hist.groupby(["customer_id", "index_code"])[
        "t_dat"
    ].agg(n_buy_hist_idx="count")
    feat["n_buy_recent_idx"] = df_trans_recent.groupby(["customer_id", "index_code"])[
        "t_dat"
    ].agg(n_buy_recent_idx="count")
    feat["days_after_buy_idx"] = df_trans_hist.groupby(["customer_id", "index_code"])[
        "t_dat"
    ].agg(days_after_buy_idx=lambda x: (ds - max(x)).days)
    feat["n_buy_hist_idxg"] = df_trans_hist.groupby(["customer_id", "index_group_no"])[
        "t_dat"
    ].agg(n_buy_hist_idxg="count")
    feat["n_buy_recent_idxg"] = df_trans_recent.groupby(
        ["customer_id", "index_group_no"]
    )["t_dat"].agg(n_buy_recent_idxg="count")
    feat["days_after_buy_idxg"] = df_trans_hist.groupby(
        ["customer_id", "index_group_no"]
    )["t_dat"].agg(days_after_buy_idxg=lambda x: (ds - max(x)).days)
    feat["n_buy_hist_sec"] = df_trans_hist.groupby(["customer_id", "section_no"])[
        "t_dat"
    ].agg(n_buy_hist_sec="count")
    feat["n_buy_recent_sec"] = df_trans_recent.groupby(["customer_id", "section_no"])[
        "t_dat"
    ].agg(n_buy_recent_sec="count")
    feat["days_after_buy_sec"] = df_trans_hist.groupby(["customer_id", "section_no"])[
        "t_dat"
    ].agg(days_after_buy_sec=lambda x: (ds - max(x)).days)
    feat["n_buy_hist_garm"] = df_trans_hist.groupby(
        ["customer_id", "garment_group_no"]
    )["t_dat"].agg(n_buy_hist_garm="count")
    feat["n_buy_recent_garm"] = df_trans_recent.groupby(
        ["customer_id", "garment_group_no"]
    )["t_dat"].agg(n_buy_recent_garm="count")
    feat["days_after_buy_garm"] = df_trans_hist.groupby(
        ["customer_id", "garment_group_no"]
    )["t_dat"].agg(days_after_buy_garm=lambda x: (ds - max(x)).days)

    for k in feat.keys():
        feat[k] = cudf.from_pandas(feat[k])

    del df_trans_yesterday, df_trans_recent, df_trans_hist, df_buy1, df_buy2, df_buy
    gc.collect()

    return feat


def add_feat(df, ds, de, dsr, der, dsh, deh, feat):
    df = cudf.from_pandas(df)
    # rate_sales_channel_hist
    df = df.merge(
        feat["rate_sales_channel_hist"],
        how="left",
        left_on=["customer_id"],
        right_index=True,
    )
    # rate_sales_channel_recent
    df = df.merge(
        feat["rate_sales_channel_recent"],
        how="left",
        left_on=["customer_id"],
        right_index=True,
    )
    # art_buy_hist
    df = df.merge(
        feat["art_buy_hist"], how="left", left_on=["article_id"], right_index=True
    )
    # art_buy_recent
    df = df.merge(
        feat["art_buy_recent"], how="left", left_on=["article_id"], right_index=True
    )
    # art_buy_yesterday
    df = df.merge(
        feat["art_buy_yesterday"], how="left", left_on=["article_id"], right_index=True
    )
    # n_buy_hist
    df = df.merge(
        feat["n_buy_hist"],
        how="left",
        left_on=["customer_id", "article_id"],
        right_index=True,
    )
    # n_buy_recent
    df = df.merge(
        feat["n_buy_recent"],
        how="left",
        left_on=["customer_id", "article_id"],
        right_index=True,
    )
    # days_after_buy
    df = df.merge(
        feat["days_after_buy"],
        how="left",
        left_on=["customer_id", "article_id"],
        right_index=True,
    )
    # n_buy_hist_all
    df = df.merge(
        feat["n_buy_hist_all"], how="left", left_on=["customer_id"], right_index=True
    )
    # n_buy_recent_all
    df = df.merge(
        feat["n_buy_recent_all"], how="left", left_on=["customer_id"], right_index=True
    )
    # days_after_buy_all
    df = df.merge(
        feat["days_after_buy_all"],
        how="left",
        left_on=["customer_id"],
        right_index=True,
    )
    # n_buy_hist_prod
    df = df.merge(
        feat["n_buy_hist_prod"],
        how="left",
        left_on=["customer_id", "product_code"],
        right_index=True,
    )
    # n_buy_recent_prod
    df = df.merge(
        feat["n_buy_recent_prod"],
        how="left",
        left_on=["customer_id", "product_code"],
        right_index=True,
    )
    # days_after_buy_prod
    df = df.merge(
        feat["days_after_buy_prod"],
        how="left",
        left_on=["customer_id", "product_code"],
        right_index=True,
    )
    # n_buy_hist_ptype
    df = df.merge(
        feat["n_buy_hist_ptype"],
        how="left",
        left_on=["customer_id", "product_type_no"],
        right_index=True,
    )
    # n_buy_recent_ptype
    df = df.merge(
        feat["n_buy_recent_ptype"],
        how="left",
        left_on=["customer_id", "product_type_no"],
        right_index=True,
    )
    # days_after_buy_ptype
    df = df.merge(
        feat["days_after_buy_ptype"],
        how="left",
        left_on=["customer_id", "product_type_no"],
        right_index=True,
    )
    # n_buy_hist_graph
    df = df.merge(
        feat["n_buy_hist_graph"],
        how="left",
        left_on=["customer_id", "graphical_appearance_no"],
        right_index=True,
    )
    # n_buy_recent_graph
    df = df.merge(
        feat["n_buy_recent_graph"],
        how="left",
        left_on=["customer_id", "graphical_appearance_no"],
        right_index=True,
    )
    # days_after_buy_graph
    df = df.merge(
        feat["days_after_buy_graph"],
        how="left",
        left_on=["customer_id", "graphical_appearance_no"],
        right_index=True,
    )
    # n_buy_hist_col
    df = df.merge(
        feat["n_buy_hist_col"],
        how="left",
        left_on=["customer_id", "colour_group_code"],
        right_index=True,
    )
    # n_buy_recent_col
    df = df.merge(
        feat["n_buy_recent_col"],
        how="left",
        left_on=["customer_id", "colour_group_code"],
        right_index=True,
    )
    # days_after_buy_col
    df = df.merge(
        feat["days_after_buy_col"],
        how="left",
        left_on=["customer_id", "colour_group_code"],
        right_index=True,
    )
    # n_buy_hist_dep
    df = df.merge(
        feat["n_buy_hist_dep"],
        how="left",
        left_on=["customer_id", "department_no"],
        right_index=True,
    )
    # n_buy_recent_dep
    df = df.merge(
        feat["n_buy_recent_dep"],
        how="left",
        left_on=["customer_id", "department_no"],
        right_index=True,
    )
    # days_after_buy_dep
    df = df.merge(
        feat["days_after_buy_dep"],
        how="left",
        left_on=["customer_id", "department_no"],
        right_index=True,
    )
    # n_buy_hist_idx
    df = df.merge(
        feat["n_buy_hist_idx"],
        how="left",
        left_on=["customer_id", "index_code"],
        right_index=True,
    )
    # n_buy_recent_idx
    df = df.merge(
        feat["n_buy_recent_idx"],
        how="left",
        left_on=["customer_id", "index_code"],
        right_index=True,
    )
    # days_after_buy_idx
    df = df.merge(
        feat["days_after_buy_idx"],
        how="left",
        left_on=["customer_id", "index_code"],
        right_index=True,
    )
    # n_buy_hist_idxg
    df = df.merge(
        feat["n_buy_hist_idxg"],
        how="left",
        left_on=["customer_id", "index_group_no"],
        right_index=True,
    )
    # n_buy_recent_idxg
    df = df.merge(
        feat["n_buy_recent_idxg"],
        how="left",
        left_on=["customer_id", "index_group_no"],
        right_index=True,
    )
    # days_after_buy_idxg
    df = df.merge(
        feat["days_after_buy_idxg"],
        how="left",
        left_on=["customer_id", "index_group_no"],
        right_index=True,
    )
    # n_buy_hist_sec
    df = df.merge(
        feat["n_buy_hist_sec"],
        how="left",
        left_on=["customer_id", "section_no"],
        right_index=True,
    )
    # n_buy_recent_sec
    df = df.merge(
        feat["n_buy_recent_sec"],
        how="left",
        left_on=["customer_id", "section_no"],
        right_index=True,
    )
    # days_after_buy_sec
    df = df.merge(
        feat["days_after_buy_sec"],
        how="left",
        left_on=["customer_id", "section_no"],
        right_index=True,
    )
    # n_buy_hist_garm
    df = df.merge(
        feat["n_buy_hist_garm"],
        how="left",
        left_on=["customer_id", "garment_group_no"],
        right_index=True,
    )
    # n_buy_recent_garm
    df = df.merge(
        feat["n_buy_recent_garm"],
        how="left",
        left_on=["customer_id", "garment_group_no"],
        right_index=True,
    )
    # days_after_buy_garm
    df = df.merge(
        feat["days_after_buy_garm"],
        how="left",
        left_on=["customer_id", "garment_group_no"],
        right_index=True,
    )
    # rebuy_rate
    df = df.merge(feat["rebuy_rate"], how="left", on="article_id")
    # ## pair
    # df["pair_art_id"] = df["article_id"].map(pairs)
    # # pair_buy_hist
    # df = df.merge(
    #     df_trans_hist.groupby(["customer_id","article_id"])["t_dat"].agg(pair_buy_hist="count").rename("").reset_index().rename(columns={"article_id":"pair_art_id"}),
    #     how="left",on=["customer_id","pair_art_id"])
    # # pair_buy_recent
    # df = df.merge(
    #     df_trans_recent.groupby(["customer_id","article_id"])["t_dat"].agg(pair_buy_recent="count").reset_index().rename(columns={"article_id":"pair_art_id"}),
    #     how="left",on=["customer_id","pair_art_id"])
    # df = df.drop(["pair_art_id"],axis=1)
    # 欠損値埋め
    df[
        [
            "n_buy_hist",
            "n_buy_recent",
            "n_buy_hist_all",
            "n_buy_recent_all",
            "n_buy_hist_prod",
            "n_buy_recent_prod",
            "n_buy_hist_ptype",
            "n_buy_recent_ptype",
            "n_buy_hist_graph",
            "n_buy_recent_graph",
            "n_buy_hist_col",
            "n_buy_recent_col",
            "n_buy_hist_dep",
            "n_buy_recent_dep",
            "n_buy_hist_idx",
            "n_buy_recent_idx",
            "n_buy_hist_idxg",
            "n_buy_recent_idxg",
            "n_buy_hist_sec",
            "n_buy_recent_sec",
            "n_buy_hist_garm",
            "n_buy_recent_garm",
            "art_buy_yesterday",
            "art_buy_recent",
            "art_buy_hist",
            "rebuy_rate",
        ]
    ] = df[
        [
            "n_buy_hist",
            "n_buy_recent",
            "n_buy_hist_all",
            "n_buy_recent_all",
            "n_buy_hist_prod",
            "n_buy_recent_prod",
            "n_buy_hist_ptype",
            "n_buy_recent_ptype",
            "n_buy_hist_graph",
            "n_buy_recent_graph",
            "n_buy_hist_col",
            "n_buy_recent_col",
            "n_buy_hist_dep",
            "n_buy_recent_dep",
            "n_buy_hist_idx",
            "n_buy_recent_idx",
            "n_buy_hist_idxg",
            "n_buy_recent_idxg",
            "n_buy_hist_sec",
            "n_buy_recent_sec",
            "n_buy_hist_garm",
            "n_buy_recent_garm",
            "art_buy_yesterday",
            "art_buy_recent",
            "art_buy_hist",
            "rebuy_rate",
        ]
    ].fillna(
        0
    )

    df[
        [
            "days_after_buy",
            "days_after_buy_all",
            "days_after_buy_prod",
            "days_after_buy_ptype",
            "days_after_buy_graph",
            "days_after_buy_col",
            "days_after_buy_dep",
            "days_after_buy_idx",
            "days_after_buy_idxg",
            "days_after_buy_sec",
            "days_after_buy_garm",
        ]
    ] = df[
        [
            "days_after_buy",
            "days_after_buy_all",
            "days_after_buy_prod",
            "days_after_buy_ptype",
            "days_after_buy_graph",
            "days_after_buy_col",
            "days_after_buy_dep",
            "days_after_buy_idx",
            "days_after_buy_idxg",
            "days_after_buy_sec",
            "days_after_buy_garm",
        ]
    ].fillna(
        10 + len_hist
    )

    df[["rate_sales_channel_hist", "rate_sales_channel_recent"]] = df[
        ["rate_sales_channel_hist", "rate_sales_channel_recent"]
    ].fillna(1.5)

    return df.to_pandas()


def recommend_train(day_start_val):
    day_start = [
        day_start_val - datetime.timedelta(days=i - 1 + len_tr) for i in tr_set
    ]
    day_end = [day_start_val - datetime.timedelta(days=i) for i in tr_set]
    day_start_rec = [x - datetime.timedelta(days=7) for x in day_start]
    day_end_rec = [x - datetime.timedelta(days=1) for x in day_start]
    day_start_hist = [x - datetime.timedelta(days=len_hist) for x in day_start]
    day_end_hist = [x - datetime.timedelta(days=1) for x in day_start]
    day_start_rec_test = day_start_val - datetime.timedelta(days=7)
    day_end_rec_test = day_start_val - datetime.timedelta(days=1)
    day_start_hist_test = day_start_val - datetime.timedelta(days=1 + len_hist)
    day_end_hist_test = day_start_val - datetime.timedelta(days=1)
    day_end_val = day_start_val + datetime.timedelta(days=6)

    day_oldest_str = day_start_hist[-1].strftime("%Y-%m-%d")
    if (
        os.path.exists(f"{cache_dir}/df_trans_{day_oldest_str}.pkl")
        and os.path.exists(f"{cache_dir}/df_art_{day_oldest_str}.pkl")
        and os.path.exists(f"{cache_dir}/df_cust_{day_oldest_str}.pkl")
    ):
        df_trans = pd.read_pickle(f"{cache_dir}/df_trans_{day_oldest_str}.pkl")
        df_art = pd.read_pickle(f"{cache_dir}/df_art_{day_oldest_str}.pkl")
        df_cust = pd.read_pickle(f"{cache_dir}/df_cust_{day_oldest_str}.pkl")
    else:
        df_trans, df_art, df_cust = read_data(day_oldest=day_start_hist[-1])

        df_trans.to_pickle(f"{cache_dir}/df_trans_{day_oldest_str}.pkl")
        df_art.to_pickle(f"{cache_dir}/df_art_{day_oldest_str}.pkl")
        df_cust.to_pickle(f"{cache_dir}/df_cust_{day_oldest_str}.pkl")

    q_date = ""
    for i in range(len(day_start)):
        if i == 0:
            q_date = f"((t_dat >= '{day_start[0]}') and (t_dat <= '{day_end[0]}'))"
        else:
            q_date = (
                q_date
                + f" or ((t_dat >= '{day_start[i]}') and (t_dat <= '{day_end[i]}'))"
            )
    top_art_all = (
        df_trans.query(q_date)
        .groupby("article_id")["t_dat"]
        .count()
        .sort_values(ascending=False)
        .index[:N]
        .tolist()
    )

    list_df_buy = []
    list_list_cust = []
    for i in range(len(day_start)):
        list_df_buy.append(
            df_trans.query(
                f"(t_dat >= '{day_start[i]}') and (t_dat <= '{day_end[i]}') and (article_id in @top_art_all)"
            )
            .drop_duplicates(["customer_id", "article_id"])[
                ["customer_id", "article_id"]
            ]
            .copy()
        )
        list_df_buy[i]["target"] = 1
        list_list_cust.append(list_df_buy[i]["customer_id"].unique().tolist())
    for iter_train in tqdm(range(n_iter)):
        list_df_nobuy = []
        list_train = []
        for i in range(len(day_start)):
            list_df_nobuy.append(
                pd.concat(
                    [
                        pd.DataFrame(
                            {
                                "customer_id": x,
                                "article_id": random.sample(top_art_all, nobuy),
                            }
                        )
                        for x in list_list_cust[i]
                    ]
                )
            )
            list_df_nobuy[i]["target"] = 0
            list_train.append(
                pd.concat([list_df_buy[i], list_df_nobuy[i]]).drop_duplicates(
                    ["customer_id", "article_id"]
                )
            )
        del list_df_nobuy

        df_train = pd.DataFrame()
        for i in tqdm(range(len(day_start))):
            feat = feat_store(
                df_trans,
                list_list_cust[i],
                day_start[i],
                day_end[i],
                day_start_rec[i],
                day_end_rec[i],
                day_start_hist[i],
                day_end_hist[i],
            )
            # 属性追加
            list_train[i] = list_train[i].merge(
                df_art[
                    [
                        "article_id",
                        "product_code",
                        "product_type_no",
                        "graphical_appearance_no",
                        "colour_group_code",
                        "department_no",
                        "index_code",
                        "index_group_no",
                        "section_no",
                        "garment_group_no",
                    ]
                ],
                how="left",
                on="article_id",
            )
            list_train[i] = list_train[i].merge(
                df_cust[
                    [
                        "customer_id",
                        "age",
                        "FN",
                        "Active",
                        "club_member_status",
                        "fashion_news_frequency",
                    ]
                ],
                how="left",
                on="customer_id",
            )
            df_train = df_train.append(
                add_feat(
                    list_train[i],
                    day_start[i],
                    day_end[i],
                    day_start_rec[i],
                    day_end_rec[i],
                    day_start_hist[i],
                    day_end_hist[i],
                    feat,
                )
            )
            del feat
        del list_train
        gc.collect()

        X_train = df_train.drop(
            [
                "customer_id",
                "product_code",
                "product_type_no",
                "department_no",
                "target",
            ],
            axis=1,
        )
        y_train = df_train["target"]
        del df_train

        list_model = []
        if n_splits == 1:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X_train, y_train, stratify=y_train
            )
            d_tr = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
            d_va = lgb.Dataset(X_va, label=y_va, free_raw_data=False)

            list_model.append(
                lgb.train(
                    lgb_params,
                    train_set=d_tr,
                    num_boost_round=n_round,
                    valid_sets=[d_tr, d_va],
                    **lgb_train_params,
                )
            )
        else:
            folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rand)
            for tr_idx, va_idx in folds.split(X_train, y_train):
                X_tr, X_va, y_tr, y_va = (
                    X_train.iloc[tr_idx],
                    X_train.iloc[va_idx],
                    y_train.iloc[tr_idx],
                    y_train.iloc[va_idx],
                )
                d_tr = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
                d_va = lgb.Dataset(X_va, label=y_va, free_raw_data=False)
                list_model.append(
                    lgb.train(
                        lgb_params,
                        train_set=d_tr,
                        num_boost_round=n_round,
                        valid_sets=[d_tr, d_va],
                        **lgb_train_params,
                    )
                )

        pd.to_pickle(list_model, f"{save_path}/models_iter{iter_train}.pkl")
        del X_train, y_train, X_tr, X_va, y_tr, y_va, d_tr, d_va
        gc.collect()

    del df_trans, df_art, df_cust
    gc.collect()
    return


def recommend_pred(series_cust, day_start_val, eval_mid=True):
    day_start = [
        day_start_val - datetime.timedelta(days=i - 1 + len_tr) for i in tr_set
    ]
    day_end = [day_start_val - datetime.timedelta(days=i) for i in tr_set]
    day_start_rec = [x - datetime.timedelta(days=7) for x in day_start]
    day_end_rec = [x - datetime.timedelta(days=1) for x in day_start]
    day_start_hist = [x - datetime.timedelta(days=len_hist) for x in day_start]
    day_end_hist = [x - datetime.timedelta(days=1) for x in day_start]
    day_start_rec_test = day_start_val - datetime.timedelta(days=7)
    day_end_rec_test = day_start_val - datetime.timedelta(days=1)
    day_start_hist_test = day_start_val - datetime.timedelta(days=1 + len_hist)
    day_end_hist_test = day_start_val - datetime.timedelta(days=1)

    day_end_val = day_start_val + datetime.timedelta(days=6)

    day_oldest_str = day_start_hist[-1].strftime("%Y-%m-%d")
    if (
        os.path.exists(f"{cache_dir}/df_trans_{day_oldest_str}.pkl")
        and os.path.exists(f"{cache_dir}/df_art_{day_oldest_str}.pkl")
        and os.path.exists(f"{cache_dir}/df_cust_{day_oldest_str}.pkl")
    ):
        df_trans = pd.read_pickle(f"{cache_dir}/df_trans_{day_oldest_str}.pkl")
        df_art = pd.read_pickle(f"{cache_dir}/df_art_{day_oldest_str}.pkl")
        df_cust = pd.read_pickle(f"{cache_dir}/df_cust_{day_oldest_str}.pkl")
    else:
        df_trans, df_art, df_cust = read_data(day_oldest=day_start_hist[-1])

        df_trans.to_pickle(f"{cache_dir}/df_trans_{day_oldest_str}.pkl")
        df_art.to_pickle(f"{cache_dir}/df_art_{day_oldest_str}.pkl")
        df_cust.to_pickle(f"{cache_dir}/df_cust_{day_oldest_str}.pkl")

    q_date = ""
    for i in range(len(day_start)):
        if i == 0:
            q_date = f"((t_dat >= '{day_start[0]}') and (t_dat <= '{day_end[0]}'))"
        else:
            q_date = (
                q_date
                + f" or ((t_dat >= '{day_start[i]}') and (t_dat <= '{day_end[i]}'))"
            )
    top_art_all = (
        df_trans.query(q_date)
        .groupby("article_id")["t_dat"]
        .count()
        .sort_values(ascending=False)
        .index[:N]
        .tolist()
    )

    list_sl = list(range(0, N, N_div))
    if list_sl[-1] != N:
        list_sl.append(N)
    df_ans = pd.DataFrame()

    feat = feat_store(
        df_trans,
        series_cust.tolist(),
        day_start_val,
        day_end_val,
        day_start_rec_test,
        day_end_rec_test,
        day_start_hist_test,
        day_end_hist_test,
    )
    del df_trans
    for iter_train in tqdm(range(n_iter)):
        df_ans_iter = pd.DataFrame()
        for iter_art in tqdm(range(len(list_sl) - 1)):
            list_model = pd.read_pickle(f"{save_path}/models_iter{iter_train}.pkl")
            top_art = top_art_all[list_sl[iter_art] : list_sl[iter_art + 1]]
            # customer_idとarticle_idの組み合わせを作成
            df_test = pd.DataFrame(
                itertools.product(series_cust.tolist(), top_art),
                columns=["customer_id", "article_id"],
            )
            # 属性追加
            df_test = df_test.merge(
                df_art[
                    [
                        "article_id",
                        "product_code",
                        "product_type_no",
                        "graphical_appearance_no",
                        "colour_group_code",
                        "department_no",
                        "index_code",
                        "index_group_no",
                        "section_no",
                        "garment_group_no",
                    ]
                ],
                how="left",
                on="article_id",
            )
            df_test = df_test.merge(
                df_cust[
                    [
                        "customer_id",
                        "age",
                        "FN",
                        "Active",
                        "club_member_status",
                        "fashion_news_frequency",
                    ]
                ],
                how="left",
                on="customer_id",
            )

            df_test = add_feat(
                df_test,
                day_start_val,
                day_end_val,
                day_start_rec_test,
                day_end_rec_test,
                day_start_hist_test,
                day_end_hist_test,
                feat,
            )
            df_test = df_test.sort_values(["customer_id", "article_id"]).reset_index(
                drop=True
            )

            df_pred = df_test[["customer_id", "article_id"]].copy()
            df_test = df_test.drop(
                ["customer_id", "product_code", "product_type_no", "department_no"],
                axis=1,
            )
            pred = np.zeros(len(df_pred))
            for i in range(n_splits):
                list_model[i].save_model(f"{save_path}/lgbm.model")
                with redirect_stdout(open(os.devnull, "w")):
                    fm = ForestInference.load(
                        filename=f"{save_path}/lgbm.model",
                        output_class=True,
                        model_type="lightgbm",
                    )
                # pred += fm.predict(df_test) / n_splits
                pred += fm.predict_proba(df_test)[:, 1] / n_splits
            df_pred["pred"] = pred

            df_ans_iter = df_ans_iter.append(df_pred)
            df_ans_iter = df_ans_iter.sort_values(
                ["customer_id", "pred"], ascending=False
            )
            df_ans_iter = df_ans_iter.groupby("customer_id").head(tmp_top)
        df_ans = df_ans.append(df_ans_iter)
        if eval_mid:
            df_ans_tmp = df_ans.copy()
            df_ans_tmp = (
                df_ans_tmp.groupby(["customer_id", "article_id"])["pred"]
                .mean()
                .reset_index()
                .sort_values(["customer_id", "pred"], ascending=False)
            )
            df_ans_tmp = df_ans_tmp.groupby("customer_id").head(12)
            df_ans_tmp["article_id"] = (
                df_ans_tmp["article_id"].astype(str).str.zfill(10)
            )

            mapk_val = mapk(
                df_agg_val_1["article_id"].tolist(),
                df_ans_tmp.groupby("customer_id")["article_id"].apply(list).tolist(),
            )
            print(f"N_iter:{iter_train+1}, mapk:{mapk_val:.5f} ")
            del df_ans_tmp
        del df_ans_iter, list_model, df_test, df_pred, pred
        gc.collect()

    df_ans = (
        df_ans.groupby(["customer_id", "article_id"])["pred"]
        .mean()
        .reset_index()
        .sort_values(["customer_id", "pred"], ascending=False)
    )
    df_ans = df_ans.groupby("customer_id").head(12)
    df_ans["article_id"] = df_ans["article_id"].astype(str).str.zfill(10)
    df_ans = df_ans.groupby("customer_id")["article_id"].apply(list).reset_index()
    df_ans = df_ans.rename({"article_id": "pred"}, axis=1)
    gc.collect()
    return df_ans


def recommend_sub():
    df_sub = pd.read_csv(path + "sample_submission.csv")

    size_block = 30000
    list_slice = list(range(0, len(df_sub), size_block))
    if list_slice[-1] != len(df_sub):
        list_slice.append(len(df_sub))
    os.makedirs(f"{save_path}/sub", exist_ok=True)
    for i in tqdm(range(len(list_slice) - 1)):
        time.sleep(1)
        if not os.path.exists(f"{save_path}/sub/submission_{i}.csv"):
            df_sub_0 = df_sub[list_slice[i] : list_slice[i + 1]].copy()
            df_ans = recommend_pred(
                df_sub_0["customer_id"],
                day_end_valtmp + datetime.timedelta(days=1),
                eval_mid=False,
            )
            df_ans["prediction"] = df_ans["pred"].apply(lambda x: " ".join(x))
            df_ans[["customer_id", "prediction"]].to_csv(
                f"{save_path}/sub/submission_{i}.csv", index=False
            )
            del df_sub_0, df_ans
            gc.collect()
    return


if mode == "cv":
    day_start_val = datetime.datetime(2020, 9, 16)
    recommend_train(day_start_val=day_start_val)

    df_trans = pd.read_csv(path + "transactions_train.csv", dtype=tran_dtypes)
    df_trans["t_dat"] = pd.to_datetime(df_trans["t_dat"], format="%Y-%m-%d")
    df_trans = df_trans.drop_duplicates(["customer_id", "article_id", "t_dat"])

    day_end_valtmp = df_trans["t_dat"].max()
    day_start_valtmp = day_end_valtmp - datetime.timedelta(days=6)
    df_trans_val_1 = df_trans.query(
        "(t_dat >= @day_start_valtmp) and (t_dat <= @day_end_valtmp)"
    ).copy()
    df_trans_val_1["article_id"] = (
        df_trans_val_1["article_id"].astype(str).str.zfill(10)
    )
    df_agg_val_1 = (
        df_trans_val_1.groupby("customer_id")["article_id"].apply(list).reset_index()
    )
    df_agg_val_1 = df_agg_val_1[df_agg_val_1["article_id"].apply(len) != 0]

    del df_trans, df_trans_val_1
    gc.collect()
    df_ans = recommend_pred(df_agg_val_1["customer_id"], day_start_valtmp)
elif mode == "sub":
    day_start_val = datetime.datetime(2020, 9, 23)
    recommend_train(day_start_val=day_start_val)
    recommend_sub()
