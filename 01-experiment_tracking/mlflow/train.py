import os
import pickle
import click
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("random-forest-train")

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


# @click.command()
# @click.option(
#     "--data_path",
#     default="./output",
#     help="Location where the processed NYC taxi trip data was saved"
# )
def run_train(data_path: str="./output"):

    print("===========Load X_train, y_train=============")
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))

    print("===========Load X_val, y_val=============")
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    with mlflow.start_run():
        mlflow.sklearn.autolog(disable=False)

        print("===========Init Random Forest Model=============")
        rf = RandomForestRegressor(max_depth=1, random_state=0)

        print("===========Fit model=============")
        rf.fit(X_train, y_train)

        print("===========Predict=============")
        y_pred = rf.predict(X_val)

        print("===========MSE=============")
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        print(f"RMSE: {rmse}")


if __name__ == '__main__':
    run_train()