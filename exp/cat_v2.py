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

from catboost import CatBoostRanker, Pool

from utils import Timer


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

dev = "gpu"
N = 12000
N_div = 75
n_iter = 1
tmp_top = 200
len_short_hist = 30
len_mid_hist = 90
len_hist = 366
n_splits = 1
n_week = 5
tr_set = [(i * 7) + 1 for i in range(n_week)]
len_tr = 7
nobuy = 50
mode = "sub"
day_start_val = datetime.datetime(2020, 9, 23)
iwata_version = "v1"
cache_dir = f"./cache/{iwata_version}"
os.makedirs(cache_dir, exist_ok=True)

RANDOM_STATE = 46
CAT_PARAMS = {
    "depth": 7,
    "learning_rate": 0.05,
    "boosting_type": "Plain",
    "bootstrap_type": "Bernoulli",
    "subsample": 0.6897509152533826,
    "reg_lambda": 0.0007386710591959062,
    "iterations": 10000,
    "od_type": "Iter",
    "od_wait": 30,
    "metric_period": 100,
    "random_seed": RANDOM_STATE,
    "task_type": "GPU",
    "gpu_ram_part": 0.95,
    "devices": "1",
    "verbose": True,
    "loss_function": "YetiRank",
    "eval_metric": "MAP:top=12",
}


t = Timer()


# https://www.kaggle.com/tkm2261/fast-pandas-left-join-357x-faster-than-pd-merge
def fast_left_join(df, join_df, on):
    return pd.concat(
        [
            df.reset_index(drop=True),
            join_df.reindex(df[on].values).reset_index(drop=True),
        ],
        axis=1,
    )


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


def rebuy_rate(df_trans_hist, attr_id="article_id"):
    df_buy1 = (
        df_trans_hist.groupby(attr_id)["customer_id"]
        .nunique()
        .reset_index()
        .rename(columns={"customer_id": "cnt_buy1"})
    )
    df_buy2 = df_trans_hist[df_trans_hist.duplicated(["customer_id", attr_id])].copy()
    df_buy2 = df_buy2.drop_duplicates(["customer_id", attr_id])
    df_buy2 = df_buy2.groupby(attr_id)[attr_id].agg(cnt_buy2="count").reset_index()
    df_buy = pd.merge(df_buy1, df_buy2, how="left", on=attr_id).fillna(0)
    df_buy["rebuy_rate"] = df_buy["cnt_buy2"] / df_buy["cnt_buy1"]
    return df_buy


def feat_store(df_trans, l_cust, ds, de, dsr, der, dsh, dsh_short, dsh_mid, deh):
    feat = {}

    df_trans_yesterday = df_trans.query("(t_dat == @der)")  # 1day
    df_trans_recent = df_trans.query("(t_dat >= @dsr) and (t_dat <= @der)")  # 1week
    df_trans_hist = df_trans.query("(t_dat >= @dsh) and (t_dat <= @deh)")  # 1year
    df_trans_hist_short = df_trans.query(
        "(t_dat >= @dsh_short) and (t_dat <= @deh)"
    )  # 1month
    df_trans_hist_mid = df_trans.query(
        "(t_dat >= @dsh_mid) and (t_dat <= @deh)"
    )  # 2month
    df_trans_hist_ch1 = df_trans_hist[df_trans_hist.sales_channel_id == 1]
    df_trans_hist_ch2 = df_trans_hist[df_trans_hist.sales_channel_id == 2]

    # Item
    feat["art_buy_hist"] = df_trans_hist.groupby(["article_id"])["t_dat"].agg(
        art_buy_hist="count"
    )
    feat["art_buy_hist_short"] = df_trans_hist_short.groupby(["article_id"])[
        "t_dat"
    ].agg(art_buy_hist_short="count")
    feat["art_buy_hist_mid"] = df_trans_hist_mid.groupby(["article_id"])["t_dat"].agg(
        art_buy_hist_mid="count"
    )
    feat["art_buy_recent"] = df_trans_recent.groupby(["article_id"])["t_dat"].agg(
        art_buy_recent="count"
    )
    feat["art_buy_yesterday"] = df_trans_yesterday.groupby(["article_id"])["t_dat"].agg(
        art_buy_yesterday="count"
    )

    df_buy = rebuy_rate(df_trans_hist, attr_id="article_id")
    df_buy = df_buy[["article_id", "rebuy_rate"]]
    df_buy.index = df_buy.article_id
    df_buy.index.name = "article_id"
    df_buy = df_buy.drop(columns=["article_id"])
    feat["rebuy_rate"] = df_buy

    # Price
    feat["art_price_hist_agg"] = (
        df_trans_hist.groupby(["article_id"])["price"]
        .agg(["mean", "median", "max", "min"])
        .add_prefix("art_price_hist_")
    )
    feat["user_price_hist_agg"] = (
        df_trans_hist.groupby(["customer_id"])["price"]
        .agg(["mean", "median", "max", "min"])
        .add_prefix("user_price_hist_")
    )

    # Code
    feat["code_buy_hist"] = df_trans_hist.groupby(["product_code"])["t_dat"].agg(
        code_buy_hist="count"
    )
    feat["code_buy_recent"] = df_trans_recent.groupby(["product_code"])["t_dat"].agg(
        code_buy_recent="count"
    )
    feat["code_buy_yesterday"] = df_trans_yesterday.groupby(["product_code"])[
        "t_dat"
    ].agg(code_buy_yesterday="count")
    df_buy = rebuy_rate(df_trans_hist, attr_id="product_code")
    df_buy = df_buy.rename(columns={"rebuy_rate": "code_rebuy_rate"})
    df_buy = df_buy[["product_code", "code_rebuy_rate"]]
    df_buy.index = df_buy.product_code
    df_buy.index.name = "product_code"
    df_buy = df_buy.drop(columns=["product_code"])
    feat["code_rebuy_rate"] = df_buy

    # Item Ch
    feat["art_buy_hist_ch1"] = df_trans_hist_ch1.groupby(["article_id"])["t_dat"].agg(
        art_buy_hist_ch1="count"
    )
    feat["art_buy_hist_ch2"] = df_trans_hist_ch2.groupby(["article_id"])["t_dat"].agg(
        art_buy_hist_ch2="count"
    )

    df_trans_yesterday = df_trans_yesterday.query("(customer_id in @l_cust)")
    df_trans_recent = df_trans_recent.query("(customer_id in @l_cust)")
    df_trans_hist = df_trans_hist.query("(customer_id in @l_cust)")
    df_trans_hist_short = df_trans_hist_short.query("(customer_id in @l_cust)")
    df_trans_hist_mid = df_trans_hist_mid.query("(customer_id in @l_cust)")

    feat["rate_sales_channel_hist"] = df_trans_hist.groupby(["customer_id"])[
        "sales_channel_id"
    ].agg(rate_sales_channel_hist="mean")
    feat["rate_sales_channel_recent"] = df_trans_recent.groupby(["customer_id"])[
        "sales_channel_id"
    ].agg(rate_sales_channel_recent="mean")

    feat["n_buy_hist"] = df_trans_hist.groupby(["customer_id", "article_id"])[
        "t_dat"
    ].agg(n_buy_hist="count")
    feat["n_buy_hist_short"] = df_trans_hist_short.groupby(
        ["customer_id", "article_id"]
    )["t_dat"].agg(n_buy_hist_short="count")
    feat["n_buy_hist_mid"] = df_trans_hist_mid.groupby(["customer_id", "article_id"])[
        "t_dat"
    ].agg(n_buy_hist_mid="count")
    feat["n_buy_recent"] = df_trans_recent.groupby(["customer_id", "article_id"])[
        "t_dat"
    ].agg(n_buy_recent="count")

    feat["days_after_buy"] = df_trans_hist.groupby(["customer_id", "article_id"])[
        "t_dat"
    ].agg(days_after_buy=lambda x: (ds - max(x)).days)

    feat["n_buy_hist_all"] = df_trans_hist.groupby(["customer_id"])["t_dat"].agg(
        n_buy_hist_all="count"
    )
    feat["n_buy_hist_short_all"] = df_trans_hist_short.groupby(["customer_id"])[
        "t_dat"
    ].agg(n_buy_hist_short_all="count")
    feat["n_buy_hist_mid_all"] = df_trans_hist_mid.groupby(["customer_id"])[
        "t_dat"
    ].agg(n_buy_hist_mid_all="count")

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
    # user * attr
    feat["n_buy_hist_ptype"] = df_trans_hist.groupby(
        ["customer_id", "product_type_no"]
    )["t_dat"].agg(n_buy_hist_ptype="count")
    feat["n_buy_recent_ptype"] = df_trans_recent.groupby(
        ["customer_id", "product_type_no"]
    )["t_dat"].agg(n_buy_recent_ptype="count")
    feat["days_after_buy_ptype"] = df_trans_hist.groupby(
        ["customer_id", "product_type_no"]
    )["t_dat"].agg(days_after_buy_ptype=lambda x: (ds - max(x)).days)
    #
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
    feat["n_buy_hist_short_dep"] = df_trans_hist_short.groupby(
        ["customer_id", "department_no"]
    )["t_dat"].agg(n_buy_hist_short_dep="count")
    feat["n_buy_hist_mid_dep"] = df_trans_hist_mid.groupby(
        ["customer_id", "department_no"]
    )["t_dat"].agg(n_buy_hist_mid_dep="count")

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
    feat["n_buy_hist_short_sec"] = df_trans_hist_short.groupby(
        ["customer_id", "section_no"]
    )["t_dat"].agg(n_buy_hist_short_sec="count")
    feat["n_buy_hist_mid_sec"] = df_trans_hist_mid.groupby(
        ["customer_id", "section_no"]
    )["t_dat"].agg(n_buy_hist_mid_sec="count")

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

    # User - attr * attr
    # index_code * colour_group_code
    feat["n_buy_hist_code_pcol"] = df_trans_hist.groupby(
        ["customer_id", "index_code", "colour_group_code"]
    )["t_dat"].agg(n_buy_hist_code_pcol="count")
    feat["n_buy_recent_code_pcol"] = df_trans_recent.groupby(
        ["customer_id", "index_code", "colour_group_code"]
    )["t_dat"].agg(n_buy_recent_code_pcol="count")
    feat["days_after_buy_code_pcol"] = df_trans_hist.groupby(
        ["customer_id", "index_code", "colour_group_code"]
    )["t_dat"].agg(days_after_buy_code_pcol=lambda x: (ds - max(x)).days)
    # index_group_no * section_no
    feat["n_buy_hist_idxg_sec"] = df_trans_hist.groupby(
        ["customer_id", "index_group_no", "section_no"]
    )["t_dat"].agg(n_buy_hist_idxg_sec="count")
    feat["n_buy_recent_idxg_sec"] = df_trans_recent.groupby(
        ["customer_id", "index_group_no", "section_no"]
    )["t_dat"].agg(n_buy_recent_idxg_sec="count")
    feat["days_after_buy_idxg_sec"] = df_trans_hist.groupby(
        ["customer_id", "index_group_no", "section_no"]
    )["t_dat"].agg(days_after_buy_idxg_sec=lambda x: (ds - max(x)).days)
    # index_group_no * garment_group_no
    feat["n_buy_hist_idxg_gramc"] = df_trans_hist.groupby(
        ["customer_id", "index_group_no", "garment_group_no"]
    )["t_dat"].agg(n_buy_hist_idxg_gram="count")
    feat["n_buy_recent_idxg_gram"] = df_trans_recent.groupby(
        ["customer_id", "index_group_no", "garment_group_no"]
    )["t_dat"].agg(n_buy_recent_idxg_gram="count")
    feat["days_after_buy_idxg_gram"] = df_trans_hist.groupby(
        ["customer_id", "index_group_no", "garment_group_no"]
    )["t_dat"].agg(days_after_buy_idxg_gram=lambda x: (ds - max(x)).days)

    if dev == "gpu":
        for k in feat.keys():
            feat[k] = cudf.from_pandas(feat[k])

    del df_trans_yesterday, df_trans_recent, df_trans_hist, df_buy
    gc.collect()

    return feat


def add_feat(df, ds, de, dsr, der, dsh, deh, feat):
    if dev == "gpu":
        df = cudf.from_pandas(df)

    # merge aid
    for col in [
        "art_buy_hist",
        "art_buy_hist_short",
        "art_buy_hist_mid",
        "art_buy_hist_ch1",
        "art_buy_hist_ch2",
        "art_buy_recent",
        "art_buy_yesterday",
        "rebuy_rate",
    ]:
        if dev == "gpu":
            df = df.merge(
                feat[col], how="left", left_on=["article_id"], right_index=True
            )
        else:
            df = fast_left_join(df, feat[col], on="article_id")
    # merge code
    for col in [
        "code_buy_hist",
        "code_buy_recent",
        "code_buy_yesterday",
        "code_rebuy_rate",
    ]:
        if dev == "gpu":
            df = df.merge(
                feat[col], how="left", left_on=["product_code"], right_index=True
            )
        else:
            df = fast_left_join(df, feat[col], on="product_code")

    # merge cid
    for col in [
        "rate_sales_channel_hist",
        "rate_sales_channel_recent",
        "n_buy_hist_all",
        "n_buy_hist_short_all",
        "n_buy_hist_mid_all",
        "n_buy_recent_all",
        "days_after_buy_all",
    ]:
        if dev == "gpu":
            df = df.merge(
                feat[col], how="left", left_on=["customer_id"], right_index=True
            )
        else:
            df = fast_left_join(df, feat[col], on="customer_id")

    # merge cid * aid
    for col in [
        "n_buy_hist",
        "n_buy_hist_short",
        "n_buy_hist_mid",
        "n_buy_recent",
        "days_after_buy",
    ]:
        if dev == "gpu":
            df = df.merge(
                feat[col],
                how="left",
                left_on=["customer_id", "article_id"],
                right_index=True,
            )
        else:
            df = fast_left_join(df, feat[col], on=["customer_id", "article_id"])

    # merge cid * code
    for col in [
        "n_buy_hist_prod",
        "n_buy_recent_prod",
        "days_after_buy_prod",
    ]:
        if dev == "gpu":
            df = df.merge(
                feat[col],
                how="left",
                left_on=["customer_id", "product_code"],
                right_index=True,
            )
        else:
            df = fast_left_join(df, feat[col], on=["customer_id", "product_code"])

    # merge cid * product_type_no
    for col in [
        "n_buy_hist_ptype",
        "n_buy_recent_ptype",
        "days_after_buy_ptype",
    ]:
        df = df.merge(
            feat[col],
            how="left",
            left_on=["customer_id", "product_type_no"],
            right_index=True,
        )

    # merge cid * graphical_appearance_no
    for col in [
        "n_buy_hist_graph",
        "n_buy_recent_graph",
        "days_after_buy_graph",
    ]:
        if dev == "gpu":
            df = df.merge(
                feat[col],
                how="left",
                left_on=["customer_id", "graphical_appearance_no"],
                right_index=True,
            )
        else:
            df = fast_left_join(
                df, feat[col], on=["customer_id", "graphical_appearance_no"]
            )

    # merge cid * colour_group_code
    for col in [
        "n_buy_hist_col",
        "n_buy_recent_col",
        "days_after_buy_col",
    ]:
        if dev == "gpu":
            df = df.merge(
                feat[col],
                how="left",
                left_on=["customer_id", "colour_group_code"],
                right_index=True,
            )
        else:
            df = fast_left_join(df, feat[col], on=["customer_id", "colour_group_code"])

    # merge cid * colour_group_code
    for col in [
        "n_buy_hist_dep",
        "n_buy_hist_short_dep",
        "n_buy_hist_mid_dep",
        "n_buy_recent_dep",
        "days_after_buy_dep",
    ]:
        if dev == "gpu":
            df = df.merge(
                feat[col],
                how="left",
                left_on=["customer_id", "department_no"],
                right_index=True,
            )
        else:
            df = fast_left_join(df, feat[col], on=["customer_id", "department_no"])

    # merge cid * index_code
    for col in [
        "n_buy_hist_idx",
        "n_buy_recent_idx",
        "days_after_buy_idx",
    ]:
        if dev == "gpu":
            df = df.merge(
                feat[col],
                how="left",
                left_on=["customer_id", "index_code"],
                right_index=True,
            )
        else:
            df = fast_left_join(df, feat[col], on=["customer_id", "index_code"])

    # merge cid * index_code
    for col in [
        "n_buy_hist_idxg",
        "n_buy_recent_idxg",
        "days_after_buy_idxg",
    ]:
        if dev == "gpu":
            df = df.merge(
                feat[col],
                how="left",
                left_on=["customer_id", "index_group_no"],
                right_index=True,
            )
        else:
            df = fast_left_join(df, feat[col], on=["customer_id", "index_group_no"])

    # merge cid * index_code
    for col in [
        "n_buy_hist_sec",
        "n_buy_hist_short_sec",
        "n_buy_hist_mid_sec",
        "n_buy_recent_sec",
        "days_after_buy_sec",
    ]:
        if dev == "gpu":
            df = df.merge(
                feat[col],
                how="left",
                left_on=["customer_id", "section_no"],
                right_index=True,
            )
        else:
            df = fast_left_join(df, feat[col], on=["customer_id", "section_no"])

    # merge cid * garment_group_no
    for col in [
        "n_buy_hist_garm",
        "n_buy_recent_garm",
        "days_after_buy_garm",
    ]:
        if dev == "gpu":
            df = df.merge(
                feat[col],
                how="left",
                left_on=["customer_id", "garment_group_no"],
                right_index=True,
            )
        else:
            df = fast_left_join(df, feat[col], on=["customer_id", "garment_group_no"])

    # merge cid * index_code * colour_group_code
    for col in [
        "n_buy_hist_code_pcol",
        "n_buy_recent_code_pcol",
        "days_after_buy_code_pcol",
    ]:
        if dev == "gpu":
            df = df.merge(
                feat[col],
                how="left",
                left_on=["customer_id", "index_code", "colour_group_code"],
                right_index=True,
            )
        else:
            df = fast_left_join(
                df, feat[col], on=["customer_id", "index_code", "colour_group_code"]
            )

    # merge cid * index_code * colour_group_code
    for col in [
        "n_buy_hist_idxg_sec",
        "n_buy_recent_idxg_sec",
        "days_after_buy_idxg_sec",
    ]:
        if dev == "gpu":
            df = df.merge(
                feat[col],
                how="left",
                left_on=["customer_id", "index_group_no", "section_no"],
                right_index=True,
            )
        else:
            df = fast_left_join(
                df, feat[col], on=["customer_id", "index_group_no", "section_no"]
            )

    # merge cid * index_code * colour_group_code
    for col in [
        "n_buy_hist_idxg_gramc",
        "n_buy_recent_idxg_gram",
        "days_after_buy_idxg_gram",
    ]:
        if dev == "gpu":
            df = df.merge(
                feat[col],
                how="left",
                left_on=["customer_id", "index_group_no", "garment_group_no"],
                right_index=True,
            )
        else:
            df = fast_left_join(
                df, feat[col], on=["customer_id", "index_group_no", "garment_group_no"]
            )

    # 欠損値埋め
    df[
        [
            "n_buy_hist",
            "n_buy_hist_short",
            "n_buy_hist_mid",
            "n_buy_recent",
            "n_buy_hist_all",
            "n_buy_hist_short_all",
            "n_buy_hist_mid_all",
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
            "n_buy_hist_short_dep",
            "n_buy_hist_mid_dep",
            "n_buy_recent_dep",
            "n_buy_hist_idx",
            "n_buy_recent_idx",
            "n_buy_hist_idxg",
            "n_buy_recent_idxg",
            "n_buy_hist_sec",
            "n_buy_recent_sec",
            "n_buy_hist_short_sec",
            "n_buy_hist_mid_sec",
            "n_buy_hist_garm",
            "n_buy_recent_garm",
            "art_buy_yesterday",
            "art_buy_recent",
            "art_buy_hist",
            "art_buy_hist_short",
            "art_buy_hist_mid",
            "art_buy_hist_ch1",
            "art_buy_hist_ch2",
            "code_buy_hist",
            "code_buy_recent",
            "code_buy_yesterday",
            "rebuy_rate",
            "code_rebuy_rate",
            "n_buy_hist_code_pcol",
            "n_buy_recent_code_pcol",
            "n_buy_hist_idxg_sec",
            "n_buy_recent_idxg_sec",
            "n_buy_hist_idxg_gram",
            "n_buy_recent_idxg_gram",
        ]
    ] = df[
        [
            "n_buy_hist",
            "n_buy_hist_short",
            "n_buy_hist_mid",
            "n_buy_recent",
            "n_buy_hist_all",
            "n_buy_hist_short_all",
            "n_buy_hist_mid_all",
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
            "n_buy_hist_short_dep",
            "n_buy_hist_mid_dep",
            "n_buy_recent_dep",
            "n_buy_hist_idx",
            "n_buy_recent_idx",
            "n_buy_hist_idxg",
            "n_buy_recent_idxg",
            "n_buy_hist_sec",
            "n_buy_recent_sec",
            "n_buy_hist_short_sec",
            "n_buy_hist_mid_sec",
            "n_buy_hist_garm",
            "n_buy_recent_garm",
            "art_buy_yesterday",
            "art_buy_recent",
            "art_buy_hist",
            "art_buy_hist_short",
            "art_buy_hist_mid",
            "art_buy_hist_ch1",
            "art_buy_hist_ch2",
            "code_buy_hist",
            "code_buy_recent",
            "code_buy_yesterday",
            "rebuy_rate",
            "code_rebuy_rate",
            "n_buy_hist_code_pcol",
            "n_buy_recent_code_pcol",
            "n_buy_hist_idxg_sec",
            "n_buy_recent_idxg_sec",
            "n_buy_hist_idxg_gram",
            "n_buy_recent_idxg_gram",
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
            "days_after_buy_code_pcol",
            "days_after_buy_idxg_sec",
            "days_after_buy_idxg_gram",
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
            "days_after_buy_code_pcol",
            "days_after_buy_idxg_sec",
            "days_after_buy_idxg_gram",
        ]
    ].fillna(
        10 + len_hist
    )

    df[["rate_sales_channel_hist", "rate_sales_channel_recent"]] = df[
        ["rate_sales_channel_hist", "rate_sales_channel_recent"]
    ].fillna(1.5)

    # ch feat
    df["art_buy_hist_ch1_x_u_hist"] = (
        df["art_buy_hist_ch1"] * df["rate_sales_channel_hist"]
    )
    df["art_buy_hist_ch1_x_u_recent"] = (
        df["art_buy_hist_ch1"] * df["rate_sales_channel_recent"]
    )
    df["art_buy_hist_ch2_x_u_hist"] = (
        df["art_buy_hist_ch2"] * df["rate_sales_channel_hist"]
    )
    df["art_buy_hist_ch3_x_u_recent"] = (
        df["art_buy_hist_ch2"] * df["rate_sales_channel_recent"]
    )

    # price
    df = df.merge(feat["art_price_hist_agg"], how="left", on="article_id")
    df = df.merge(feat["user_price_hist_agg"], how="left", on="customer_id")
    df["diff_price_hist_median_median"] = (
        df["art_price_hist_median"] - df["user_price_hist_median"]
    )
    df["diff_price_hist_median_max"] = (
        df["art_price_hist_median"] - df["user_price_hist_max"]
    )
    df["diff_price_hist_median_min"] = (
        df["art_price_hist_median"] - df["user_price_hist_min"]
    )
    df["diff_price_hist_max_max"] = df["art_price_hist_max"] - df["user_price_hist_max"]
    df["diff_price_hist_max_median"] = (
        df["art_price_hist_max"] - df["user_price_hist_median"]
    )
    df["diff_price_hist_min_min"] = df["art_price_hist_min"] - df["user_price_hist_min"]
    df["diff_price_hist_min_median"] = (
        df["art_price_hist_min"] - df["user_price_hist_median"]
    )

    # to rate
    for col in [
        "art_buy",
        "n_buy",
        "rate_sales_channel",
    ]:
        df[f"{col}_recent_hist_rate"] = df[f"{col}_recent"] / df[f"{col}_hist"]
    for col in ["ptype", "graph", "col", "dep", "idx", "idxg", "sec", "garm"]:
        df[f"n_buy_rate_{col}"] = df[f"n_buy_recent_{col}"] / df[f"n_buy_hist_{col}"]

    if dev == "gpu":
        df = df.to_pandas()
    return df


def recommend_train(day_start_val):

    day_start = [
        day_start_val - datetime.timedelta(days=i - 1 + len_tr) for i in tr_set
    ]
    day_end = [day_start_val - datetime.timedelta(days=i) for i in tr_set]
    day_start_rec = [x - datetime.timedelta(days=7) for x in day_start]
    day_end_rec = [x - datetime.timedelta(days=1) for x in day_start]
    day_start_hist = [x - datetime.timedelta(days=len_hist) for x in day_start]
    day_end_hist = [x - datetime.timedelta(days=1) for x in day_start]
    day_start_hist_short = [
        x - datetime.timedelta(days=len_short_hist) for x in day_start
    ]
    day_start_hist_mid = [x - datetime.timedelta(days=len_mid_hist) for x in day_start]

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

    df_art.index = df_art.article_id
    df_art.index.name = "article_id"
    df_art = df_art.drop(columns=["article_id"])

    df_cust.index = df_cust.customer_id
    df_cust.index.name = "customer_id"
    df_cust = df_cust.drop(columns=["customer_id"])

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
        list_df_buy[i]["week"] = i
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
            list_df_nobuy[i]["week"] = i
            list_train.append(
                pd.concat([list_df_buy[i], list_df_nobuy[i]]).drop_duplicates(
                    ["customer_id", "article_id"]
                )
            )
        del list_df_nobuy

        df_train = pd.DataFrame()
        for i in tqdm(range(len(day_start))):

            with t.timer(f"feat_store {iter_train} {i}"):
                feat = feat_store(
                    df_trans,
                    list_list_cust[i],
                    day_start[i],
                    day_end[i],
                    day_start_rec[i],
                    day_end_rec[i],
                    day_start_hist[i],
                    day_start_hist_short[i],
                    day_start_hist_mid[i],
                    day_end_hist[i],
                )

            list_train[i] = fast_left_join(
                list_train[i],
                df_art[
                    [
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
                on="article_id",
            )
            list_train[i] = fast_left_join(
                list_train[i],
                df_cust[
                    [
                        "age",
                        "FN",
                        "Active",
                        "club_member_status",
                        "fashion_news_frequency",
                    ]
                ],
                on="customer_id",
            )

            df_add_feat = add_feat(
                list_train[i],
                day_start[i],
                day_end[i],
                day_start_rec[i],
                day_end_rec[i],
                day_start_hist[i],
                day_end_hist[i],
                feat,
            )
            if i == 0:
                df_val = df_add_feat
            else:
                df_train = df_train.append(df_add_feat)
            del feat
        del list_train
        gc.collect()

        df_train.to_pickle(f"{save_path}/df_train_iter{iter_train}.pkl")
        df_val.to_pickle(f"{save_path}/df_val_iter{iter_train}.pkl")

        cat_features_index = []
        df_train = df_train.sort_values(["customer_id", "week"])
        X_train = df_train.drop(
            [
                "customer_id",
                "article_id",
                "product_code",
                "product_type_no",
                "department_no",
                "week",
                "target",
            ],
            axis=1,
        )
        y_train = df_train["target"]
        idx = df_train.groupby(["customer_id", "week"])["customer_id"].count().values
        group_id_train = [
            i for i, bascket_num in enumerate(idx) for _ in range(bascket_num)
        ]

        df_val = df_val.sort_values(["customer_id", "week"])
        X_val = df_val.drop(
            [
                "customer_id",
                "article_id",
                "product_code",
                "product_type_no",
                "department_no",
                "week",
                "target",
            ],
            axis=1,
        )
        y_val = df_val["target"]
        idx = df_val.groupby(["customer_id", "week"])["customer_id"].count().values
        group_id_val = [
            i for i, bascket_num in enumerate(idx) for _ in range(bascket_num)
        ]

        train_pool = Pool(
            data=X_train,
            label=y_train,
            group_id=group_id_train,
            cat_features=cat_features_index,
        )
        val_pool = Pool(
            data=X_val,
            label=y_val,
            group_id=group_id_val,
            cat_features=cat_features_index,
        )

        list_model = []
        model = CatBoostRanker(**CAT_PARAMS)
        model.fit(train_pool, eval_set=val_pool)
        list_model.append(model)
        pd.to_pickle(list_model, f"{save_path}/models_iter{iter_train}.pkl")
        del X_train, y_train, X_val, y_val, train_pool, val_pool
        gc.collect()

    del df_trans, df_art, df_cust
    gc.collect()
    return


def recommend_pred(series_cust, day_start_val, eval_mid=True, sub_i=-1):
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
    day_start_hist_short_test = day_start_val - datetime.timedelta(
        days=1 + len_short_hist
    )
    day_start_hist_mid_test = day_start_val - datetime.timedelta(days=1 + len_mid_hist)
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

    df_art.index = df_art.article_id
    df_art.index.name = "article_id"
    df_art = df_art.drop(columns=["article_id"])

    df_cust.index = df_cust.customer_id
    df_cust.index.name = "customer_id"
    df_cust = df_cust.drop(columns=["customer_id"])

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

    with t.timer(f"feat_store (pred)"):
        feat = feat_store(
            df_trans,
            series_cust.tolist(),
            day_start_val,
            day_end_val,
            day_start_rec_test,
            day_end_rec_test,
            day_start_hist_test,
            day_start_hist_short_test,
            day_start_hist_mid_test,
            day_end_hist_test,
        )
    del df_trans
    for iter_train in tqdm(range(n_iter)):
        df_ans_iter = pd.DataFrame()
        df_pred_list = []
        list_model = pd.read_pickle(f"{save_path}/models_iter{iter_train}.pkl")
        for iter_art in tqdm(range(len(list_sl) - 1)):
            top_art = top_art_all[list_sl[iter_art] : list_sl[iter_art + 1]]

            df_test = pd.DataFrame(
                itertools.product(series_cust.tolist(), top_art),
                columns=["customer_id", "article_id"],
            )

            df_test = fast_left_join(
                df_test,
                df_art[
                    [
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
                on="article_id",
            )

            df_test = fast_left_join(
                df_test,
                df_cust[
                    [
                        "age",
                        "FN",
                        "Active",
                        "club_member_status",
                        "fashion_news_frequency",
                    ]
                ],
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

            df_pred = df_test[["customer_id", "article_id"]].copy()
            df_test = df_test.drop(
                [
                    "customer_id",
                    "article_id",
                    "product_code",
                    "product_type_no",
                    "department_no",
                ],
                axis=1,
            )

            pred = np.zeros(len(df_pred))
            for i in range(n_splits):
                pred += list_model[i].predict(df_test) / n_splits
                # if dev == "gpu":
                #     with redirect_stdout(open(os.devnull, "w")):
                #         fm = ForestInference.load(
                #             filename=f"{save_path}/lgbm.model",
                #             output_class=True,
                #             model_type="lightgbm",
                #         )
                #     pred += fm.predict_proba(df_test)[:, 1] / n_splits
                # else:
                #     pred += list_model[i].predict(df_test) / n_splits

            df_pred["pred"] = pred

            df_pred_list.append(df_pred)
            if iter_art % 20 == 0:
                df_ans_iter_list = pd.concat(df_pred_list)
                df_ans_iter = pd.concat([df_ans_iter, df_ans_iter_list])
                df_ans_iter = df_ans_iter.sort_values(
                    ["customer_id", "pred"], ascending=False
                )
                df_ans_iter = df_ans_iter.groupby("customer_id").head(tmp_top)
                df_pred_list = []

            if eval_mid and iter_art % 20 == 0:
                df_ans_tmp = df_ans_iter.copy()
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
                    df_ans_tmp.groupby("customer_id")["article_id"]
                    .apply(list)
                    .tolist(),
                )
                print(f"topN: ~{list_sl[iter_art + 1]}, mapk:{mapk_val:.5f} ")
                del df_ans_tmp

        df_ans_iter_list = pd.concat(df_pred_list)
        df_ans_iter = pd.concat([df_ans_iter, df_ans_iter_list])
        df_ans_iter = df_ans_iter.sort_values(["customer_id", "pred"], ascending=False)
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
    df_ans.to_pickle(f"{save_path}/df_ans_eval_mid{int(eval_mid)}_{sub_i}.pkl")

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
                sub_i=i,
            )
            df_ans["prediction"] = df_ans["pred"].apply(lambda x: " ".join(x))
            df_ans[["customer_id", "prediction"]].to_csv(
                f"{save_path}/sub/submission_{i}.csv", index=False
            )
            del df_sub_0, df_ans
            gc.collect()
    return


day_start_val_str = day_start_val.strftime("%Y-%m-%d")
if os.path.exists(f"{cache_dir}/df_agg_val_1_{day_start_val_str}.pkl"):
    day_start_valtmp = datetime.datetime(2020, 9, 16)
    day_end_valtmp = day_start_valtmp + datetime.timedelta(days=6)
    df_agg_val_1 = pd.read_pickle(f"{cache_dir}/df_agg_val_1_{day_start_val_str}.pkl")
else:
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
    df_agg_val_1.to_pickle(f"{cache_dir}/df_agg_val_1_{day_start_val_str}.pkl")
    del df_trans, df_trans_val_1
    gc.collect()

if mode == "cv":
    # recommend_train(day_start_val=day_start_val)
    recommend_pred(df_agg_val_1["customer_id"], day_start_valtmp)

elif mode == "sub":
    day_start_val = datetime.datetime(2020, 9, 23)
    # recommend_train(day_start_val=day_start_val)
    recommend_sub()
