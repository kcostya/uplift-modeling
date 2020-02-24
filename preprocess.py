from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split


def fix_age(df_clients, real_fix=True):
    # create a copy of age column
    df_clients["age2"] = df_clients["age"]

    age_index = (df_clients["age"] < -900) & (df_clients["age"] > -1000)
    df_clients.loc[age_index, "age2"] = -1 * df_clients.loc[age_index, "age"] + 1019
    df_clients.loc[age_index, "age_antinorm"] = 1

    age_index = (df_clients["age"] > 900) & (df_clients["age"] < 1000)
    df_clients.loc[age_index, "age2"] = 1019 - df_clients.loc[age_index, "age"]
    df_clients.loc[age_index, "age_antinorm"] = 2

    age_index = (df_clients["age"] > 1900) & (df_clients["age"] < 2000)
    df_clients.loc[age_index, "age2"] = 2019 - df_clients.loc[age_index, "age"]
    df_clients.loc[age_index, "age_antinorm"] = 3

    age_index = (df_clients["age"] > 120) & (df_clients["age"] < 200)
    df_clients.loc[age_index, "age2"] = df_clients.loc[age_index, "age"] - 100
    df_clients.loc[age_index, "age_antinorm"] = 4

    age_index = (df_clients["age"] > 1800) & (df_clients["age"] < 1900)
    df_clients.loc[age_index, "age2"] = df_clients.loc[age_index, "age"] - 1800
    df_clients.loc[age_index, "age_antinorm"] = 5

    # use the modified copy
    if real_fix:
        df_clients["age"] = df_clients["age2"]

    df_clients.drop("age2", axis=1, inplace=True)

    return df_clients


def predict_broke_age(df):
    """Это функция тренирует модель на предсказание возраста и делает предсказание ошибочного возраста.
    Под ошибочным понимается тот, который выбивается из интервала 14-95 лет"""

    broke_age_index = df[(df["age"] < 14) | (df["age"] > 95)].index

    X = df[~df.index.isin(broke_age_index)].drop(
        ["age", "gender", "first_issue_date", "first_redeem_date"], axis=1
    )
    y = df[~df.index.isin(broke_age_index)]["age"]

    params = {
        "n_jobs": -1,
        "seed": 42,
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "rmse",
        "n_estimators": 5000,
        "verbose": -1,
    }

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=42)

    model = LGBMRegressor(**params)

    model.fit(
        X_train,
        y_train,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=100,
        verbose=False,
    )

    age_predict = model.predict(
        df.loc[broke_age_index, :].drop(
            ["age", "gender", "first_issue_date", "first_redeem_date"], axis=1
        )
    )

    df.loc[broke_age_index, "age"] = age_predict.astype(int)

    return df
