# models/prophet_forecast.py
import pickle
import pandas as pd
import os

class ProphetForecaster:
    def __init__(self, model_dir="models"):
        model_p = f"{model_dir}/model_prophet.pkl"
        if not os.path.exists(model_p):
            raise FileNotFoundError("model_prophet.pkl missing.")
        self.model = pickle.load(open(model_p, "rb"))

    def predict(self, dates):
        df = pd.DataFrame({"ds": dates})
        fc = self.model.predict(df)
        return fc["yhat"].values


def predict_prophet(dates, model_dir="models"):
    return ProphetForecaster(model_dir).predict(dates)
