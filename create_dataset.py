import pandas as pd
from tqdm import tqdm

from feature_generator import (
    get_date_features,
    get_gender_features,
    get_purchase_features,
)
from preprocess import fix_age
from utils import timer


def create_dataset():
    # read data
    with timer("loading clients and product data"):
        df_clients = pd.read_csv("data/clients.csv", index_col="client_id")
        df_products = pd.read_csv("data/products.csv")

    df_products.fillna("None", inplace=True)
    df_products.loc[df_products["netto"] == "None", "netto"] = 0

    with timer("loading purchases data"):
        purchases_chunks = pd.read_csv("data/purchases.csv", chunksize=10 ** 6)
        data = []

        for chunk in tqdm(purchases_chunks):
            chunk.fillna(0, inplace=True)
            chunk["hour"] = pd.to_datetime(chunk["transaction_datetime"]).dt.hour
            chunk["dayofweek"] = pd.to_datetime(
                chunk["transaction_datetime"]
            ).dt.dayofweek
            data += get_purchase_features(chunk, df_products)

        print("creating purchases dataframe...")
        df_purchases = pd.DataFrame(data)

    # feature extraction from clients data
    df_clients = get_date_features(df_clients)
    df_clients = get_gender_features(df_clients)
    df_clients = df_clients.fillna(value=-999)
    df_clients = fix_age(df_clients, real_fix=True)

    with timer("merge clients and purchases dataframes"):
        df_features = pd.merge(
            df_clients.reset_index(), df_purchases, how="inner"
        ).set_index("client_id")

    duplicate_index = df_features.index.duplicated()
    df_features = df_features.loc[~duplicate_index, :]

    with timer("saving data to csv"):
        df_features.to_csv("data/df_features.csv", index="client_id")


if __name__ == "__main__":
    create_dataset()
