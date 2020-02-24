from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd


def get_purchase_features(df_purchases, df_products):
    """Get features from purchases information grouped by clients."""

    dict_product = dict(
        zip(df_products["product_id"].values, df_products.iloc[:, 1:6].values)
    )
    dict_product_number = dict(
        zip(df_products["product_id"].values, df_products.iloc[:, -3:].values)
    )

    features_product = features_dict(df_products)
    features_product["netto"] = 0
    features_product["trademark"] = 0
    features_product["alcohol"] = 0
    features_product["regular_points_received"] = 0
    features_product["express_points_received"] = 0
    features_product["regular_points_spent"] = 0
    features_product["express_points_spent"] = 0

    for i in range(24):
        if i < 7:
            features_product["dayofweek_" + str(i)] = 0
        features_product["hour_" + str(i)] = 0

    points_list = [
        "regular_points_received",
        "express_points_received",
        "regular_points_spent",
        "express_points_spent",
    ]

    data = []

    for i, (id_c, client) in enumerate(df_purchases.groupby("client_id", sort=False)):
        features = features_product.copy()
        n_trans = client["transaction_id"].nunique()
        features["transactions"] = n_trans
        features["sum"] = client["trn_sum_from_iss"].sum()
        features["trn_sum_from_red"] = client["trn_sum_from_red"].sum()
        features["n_store"] = client["store_id"].nunique()
        features["n_product"] = client["product_id"].nunique()
        features["max_price"] = client["trn_sum_from_iss"].max()
        features["min_price"] = client["trn_sum_from_iss"].min()
        features["median_price"] = client["trn_sum_from_iss"].median()
        features["std_price"] = client["trn_sum_from_iss"].std()
        features["quantity"] = client["product_quantity"].sum()
        features["first_buy_sum"] = client["purchase_sum"].iloc[0]
        features["last_buy_sum"] = client["purchase_sum"].iloc[-1]
        try:
            features["almost_last_buy"] = client["purchase_sum"].unique()[-2]
        except:
            features["almost_last_buy"] = client["purchase_sum"].unique()[0]

        features["client_id"] = client["client_id"].iloc[0]
        features["transaction_max_delay"] = transaction_max_delay(client)

        # features from products
        count_products = Counter(client["product_id"])
        for product in count_products.keys():
            values = dict_product[product]
            for value in values:
                if type(value) != str:
                    features["segment_id_" + str(value)] += count_products[product]
                else:
                    features[f"product_id_{value}"] = count_products[product]

        temp_dict_quantity = dict(zip(client["product_id"], client["product_quantity"]))

        for product, quantity in temp_dict_quantity.items():
            features["netto"] += quantity * dict_product_number[product][0]
            features["trademark"] += quantity * dict_product_number[product][1]
            features["alcohol"] += quantity * dict_product_number[product][2]

        # Features from date
        temp_dict_date = dict(
            zip(client["transaction_id"].values, client["dayofweek"].values)
        )
        for dayofweek in temp_dict_date.values():
            features["dayofweek_" + str(dayofweek)] += 1

        temp_dict_date = dict(
            zip(client["transaction_id"].values, client["hour"].values)
        )
        for hour in temp_dict_date.values():
            features["hour_" + str(hour)] += 1

        # Features from points
        points_dict = dict(
            zip(client["transaction_id"].values, client[points_list].values)
        )
        for point in points_dict.values():
            features["regular_points_received"] += point[0]
            features["express_points_received"] += point[1]
            features["regular_points_spent"] += point[2]
            features["express_points_spent"] += point[3]

        # Average features
        features["avg_regular_points_received"] = (
            features["regular_points_received"] / n_trans
        )
        features["avg_express_points_received"] = (
            features["express_points_received"] / n_trans
        )
        features["avg_regular_points_spent"] = (
            features["regular_points_spent"] / n_trans
        )
        features["avg_express_points_spent"] = (
            features["express_points_spent"] / n_trans
        )
        features["avg_sum_from_red"] = features["trn_sum_from_red"] / n_trans
        features["avg_price_product"] = features["sum"] / n_trans
        features["avg_delay_beetwen_transc"] = (
            features["transaction_max_delay"] / n_trans
        )
        features["avg_sum"] = features["sum"] / n_trans
        features["avg_quantity"] = features["quantity"] / n_trans
        features["avg_netto"] = features["netto"] / n_trans
        features["avg_trademark"] = features["trademark"] / n_trans
        features["avg_alcohol"] = features["alcohol"] / n_trans

        data.append(features)

    return data


def features_dict(df_products):
    """Функция создания словаря для создания признаков на основе купленных продуктов"""
    features_product = dict(
        set(
            zip(
                "level_1_" + df_products["level_1"].unique(),
                np.zeros(len(df_products["level_1"].unique())),
            )
        )
    )
    for col in df_products.columns[2:6]:
        if col == "segment_id":
            features_product.update(
                dict(
                    set(
                        zip(
                            "segment_id_" + (df_products[col].astype(str).unique()),
                            np.zeros(len(df_products[col].unique())),
                        )
                    )
                )
            )
        else:
            features_product.update(
                dict(
                    set(
                        zip(
                            f"{col}_" + df_products[col].unique(),
                            np.zeros(len(df_products[col].unique())),
                        )
                    )
                )
            )

    return features_product


def transaction_max_delay(client):
    """Функция для подсчета дней между первой и последней покупкой, где:
    client - данные по одному клиенту"""

    first_transaction = pd.to_datetime(client["transaction_datetime"].iloc[0])
    last_transaction = pd.to_datetime(client["transaction_datetime"].iloc[-1])
    day_delay = (last_transaction - first_transaction).days
    return day_delay


def get_date_features(df):
    df["first_issue_date"] = pd.to_datetime(df["first_issue_date"])
    df["first_redeem_date"] = pd.to_datetime(df["first_redeem_date"])

    df["first_issue_unixtime"] = (df["first_issue_date"]).astype(int) / 10 ** 9
    df["first_redeem_unixtime"] = (df["first_redeem_date"]).astype(int) / 10 ** 9
    df["issue_redeem_delay"] = df["first_redeem_unixtime"] - df["first_issue_unixtime"]

    df["first_redeem_date"] = df["first_redeem_date"].fillna(
        datetime(2019, 3, 19, 0, 0)
    )

    df["first_issue_date_weekday"] = df["first_issue_date"].dt.weekday
    df["first_redeem_date_weekday"] = df["first_redeem_date"].dt.weekday
    df["first_issue_date_hour"] = df["first_issue_date"].dt.hour
    df["first_redeem_date_hour"] = df["first_redeem_date"].dt.hour

    df["redeem_date_mo"] = df["first_redeem_date"].dt.month
    df["redeem_date_week"] = df["first_redeem_date"].dt.week
    df["redeem_date_doy"] = df["first_redeem_date"].dt.dayofyear
    df["redeem_date_q"] = df["first_redeem_date"].dt.quarter
    df["redeem_date_ms"] = df["first_redeem_date"].dt.is_month_start
    df["redeem_date_me"] = df["first_redeem_date"].dt.is_month_end

    df.drop(columns=["first_issue_date", "first_redeem_date"], inplace=True)

    return df


def get_gender_features(df):
    df["gender_M"] = (df["gender"] == "M").astype(int)
    df["gender_F"] = (df["gender"] == "F").astype(int)
    df["gender_U"] = (df["gender"] == "U").astype(int)
    df.drop(["gender"], axis=1, inplace=True)
    return df
