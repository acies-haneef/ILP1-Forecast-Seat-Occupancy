# models/polynomial_forecast.py
import pickle
import os
import pandas as pd

class PolynomialForecaster:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.scaler = None
        self.poly = None
        self.model = None
        self._load()

    def _load(self):
        scaler_p = f"{self.model_dir}/scaler.pkl"
        poly_p   = f"{self.model_dir}/poly.pkl"
        model_p  = f"{self.model_dir}/model_poly.pkl"

        if not all([os.path.exists(scaler_p), os.path.exists(poly_p), os.path.exists(model_p)]):
            raise FileNotFoundError("Polynomial model files missing in models folder.")

        self.scaler = pickle.load(open(scaler_p,"rb"))
        self.poly   = pickle.load(open(poly_p,"rb"))
        self.model  = pickle.load(open(model_p,"rb"))

    def predict(self, df, cols):
        X = df[cols].astype(float).fillna(0)
        X_scaled = self.scaler.transform(X)
        X_poly = self.poly.transform(X_scaled)
        return self.model.predict(X_poly)


def predict_poly(df, cols, model_dir="models"):
    return PolynomialForecaster(model_dir).predict(df, cols)

def train_poly_model(df, target_col, feature_cols):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression

    X = df[feature_cols].values
    y = df[target_col].values

    poly = PolynomialFeatures(degree=1, include_bias=False)  # same as your model
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    feature_names = poly.get_feature_names_out(feature_cols)

    return model, X_poly, feature_names
