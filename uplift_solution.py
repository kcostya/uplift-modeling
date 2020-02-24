import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

from metrics import uplift_score
from train import uplift_fit_predict
from utils import timer

# config
lgbm_params = {
    "learning_rate": 0.01,
    "max_depth": 6,
    "num_leaves": 20,
    "min_data_in_leaf": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.01,
    "max_bin": 416,
    "bagging_freq": 3,
    "reg_lambda": 0.01,
    "n_estimators": 600,
    "application": "binary",
}

# read data
print("start loading data...")
with timer("loading data"):
    df_train = pd.read_csv("data/uplift_train.csv", index_col="client_id")
    df_test = pd.read_csv("data/uplift_test.csv", index_col="client_id")
    df_features = pd.read_csv("data/df_features.csv", index_col="client_id")

drop_features = [
    "first_issue_date",
    "first_redeem_date",
    "issue_redeem_delay",
    "first_issue_date_hour",
    "age_antinorm",
]

df_features = df_features.drop(columns=drop_features)

# drop duplicated indexes
duplicate_index = df_features.index.duplicated()
df_features = df_features.loc[~duplicate_index, :]

# cross-validation
indices_train = df_train.index

indices_test = df_test.index
indices_learn, indices_valid = train_test_split(
    indices_train, test_size=0.3, random_state=123
)

print("start training uplift models...")
with timer("train uplift models"):
    valid_uplift = uplift_fit_predict(
        model=LGBMClassifier(**lgbm_params),
        X_train=df_features.loc[indices_learn, :].fillna(0).values,
        treatment_train=df_train.loc[indices_learn, "treatment_flg"].values,
        target_train=df_train.loc[indices_learn, "target"].values,
        X_test=df_features.loc[indices_valid, :].fillna(0).values,
    )

valid_score = uplift_score(
    valid_uplift,
    treatment=df_train.loc[indices_valid, "treatment_flg"].values,
    target=df_train.loc[indices_valid, "target"].values,
)
print(f"Validation score: {valid_score:.4f}")

# predict test
with timer("predict test set"):
    test_uplift = uplift_fit_predict(
        model=LGBMClassifier(**lgbm_params),
        X_train=df_features.loc[indices_train, :].fillna(0).values,
        treatment_train=df_train.loc[indices_train, "treatment_flg"].values,
        target_train=df_train.loc[indices_train, "target"].values,
        X_test=df_features.loc[indices_test, :].fillna(0).values,
    )

df_submission = pd.DataFrame({"uplift": test_uplift}, index=df_test.index)
df_submission.to_csv(f"submissions/submission_{valid_score:.4f}.csv")
