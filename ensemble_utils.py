"""
ensemble_utils.py
-----------------
Helper code shared by the notebook and the Streamlit dashboard.

Author : Serasinghe S T H C H (224179H)
Project: Ensemble learning on Sri Lankan used-car prices

Everything the saved .pkl models need at prediction time (feature
engineering + the RF+ANN hybrid class) lives here, so the pickles can be
loaded anywhere as long as this file is importable.
"""
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REFERENCE_YEAR = 2025          # listings were scraped Dec-2024 to Feb-2025
SEED = 42

# raw columns that the user (or the dashboard) has to provide
RAW_COLS = ["Brand", "Model", "YOM", "Engine_cc", "Gear", "Fuel", "Town",
            "Leasing", "Condition", "AC", "PowerSteering", "PowerMirror",
            "PowerWindow"]

CAT_COLS = ["Brand", "Model", "Town", "Gear", "Fuel", "Leasing", "Condition"]
NUM_COLS = ["Age", "Engine_cc", "AC", "PowerSteering", "PowerMirror",
            "PowerWindow", "EquipScore"]


# ---------------------------------------------------------------------------
# Data loading / cleaning
# ---------------------------------------------------------------------------
def load_and_clean(path):
    """Read the Kaggle csv and return a tidy dataframe (Price in LKR lakhs)."""
    df = pd.read_csv(path)
    df = df.rename(columns={
        "Engine (cc)": "Engine_cc", "Fuel Type": "Fuel",
        "AIR CONDITION": "AC", "POWER STEERING": "PowerSteering",
        "POWER MIRROR": "PowerMirror", "POWER WINDOW": "PowerWindow",
        "Millage(KM)": "Mileage_km",
    })
    return df


def add_features(X):
    """
    Feature engineering that is applied *inside* every pipeline.
    Input : raw columns (see RAW_COLS)
    Output: CAT_COLS + NUM_COLS
    """
    X = pd.DataFrame(X).copy()
    X["Age"] = (REFERENCE_YEAR - X["YOM"].astype(float)).clip(lower=0)
    for c in ["AC", "PowerSteering", "PowerMirror", "PowerWindow"]:
        X[c] = (X[c].astype(str).str.lower() == "available").astype(int)
    X["EquipScore"] = X[["AC", "PowerSteering", "PowerMirror", "PowerWindow"]].sum(axis=1)
    X["Engine_cc"] = X["Engine_cc"].astype(float)
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
    return X[CAT_COLS + NUM_COLS]


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------
def make_preprocessor(scale=False, min_freq=10):
    """
    One-hot encode the categoricals (rare levels are pooled into one
    'infrequent' bucket so the 1500+ car models do not explode the width)
    and optionally standardise the numeric columns (needed for SVR / ANN,
    NOT needed for trees).
    """
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    return ColumnTransformer([
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  min_frequency=min_freq, sparse_output=False)),
        ]), CAT_COLS),
        ("num", Pipeline(num_steps), NUM_COLS),
    ])


def make_pipe(estimator, scale=False, log_target=True):
    """
    raw columns -> feature engineering -> preprocessing -> model.
    The target is modelled as log(1+price) and converted back automatically,
    so .predict() always returns the price in LKR lakhs.
    """
    pipe = Pipeline([
        ("fe", FunctionTransformer(add_features)),
        ("pre", make_preprocessor(scale=scale)),
        ("model", estimator),
    ])
    if not log_target:
        return pipe
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p,
                                      inverse_func=np.expm1)


def native_cat_preprocessor():
    """Pass raw categorical strings straight through (for CatBoost)."""
    return ColumnTransformer([
        ("cat", "passthrough", CAT_COLS),
        ("num", SimpleImputer(strategy="median"), NUM_COLS),
    ]).set_output(transform="pandas")


class NativeCatBoost(BaseEstimator, RegressorMixin):
    """
    Thin wrapper so CatBoost can use its *native* categorical handling
    (ordered target statistics) inside an sklearn pipeline. CatBoost's own
    `cat_features` argument does not survive sklearn's clone(), hence this.
    Expects the DataFrame produced by native_cat_preprocessor().
    """

    def __init__(self, iterations=800, learning_rate=0.05, depth=6,
                 l2_leaf_reg=3.0, random_state=SEED):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_state = random_state

    def _cat_idx(self, X):
        return [i for i, c in enumerate(X.columns) if c.split("__")[-1] in CAT_COLS]

    def fit(self, X, y):
        from catboost import CatBoostRegressor
        self.model_ = CatBoostRegressor(
            iterations=self.iterations, learning_rate=self.learning_rate,
            depth=self.depth, l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_state, verbose=0, thread_count=-1)
        self.model_.fit(X, y, cat_features=self._cat_idx(X))
        return self

    def predict(self, X):
        return self.model_.predict(X)


def make_native_catboost_pipe(**kwargs):
    pipe = Pipeline([
        ("fe", FunctionTransformer(add_features)),
        ("pre", native_cat_preprocessor()),
        ("model", NativeCatBoost(**kwargs)),
    ])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p,
                                      inverse_func=np.expm1)


# ---------------------------------------------------------------------------
# Hybrid model: Random Forest + ANN
# ---------------------------------------------------------------------------
class RFANNHybrid(BaseEstimator, RegressorMixin):
    """
    Hybrid of a Random Forest (bagging) and an ANN (MLP), three flavours:

    mode="blend"      RF and ANN are trained independently on log(price);
                      prediction = w * RF + (1 - w) * ANN.
    mode="residual"   ANN is trained on the RF's *out-of-bag* residuals
                      (true - OOB prediction) and added back on top of the RF.
                      OOB predictions are honest because each tree never saw
                      that row, so there is no leakage.
    mode="rf_feature" ANN gets the scaled features PLUS the RF's OOB prediction
                      as an extra input (RF output feeds the ANN).

    Works on the log(price) scale; wrap with make_hybrid_pipe() to get lakhs.
    """

    def __init__(self, mode="blend", n_estimators=300, max_features=0.33,
                 min_samples_leaf=1, hidden_layer_sizes=(64, 32), alpha=1e-3,
                 rf_weight=0.5, ann_weight=1.0, n_ann=1, random_state=SEED):
        self.mode = mode
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.rf_weight = rf_weight
        self.ann_weight = ann_weight
        self.n_ann = n_ann
        self.random_state = random_state

    def _new_ann(self, seed):
        return MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes, alpha=self.alpha,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=15,
            max_iter=400, random_state=seed)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y, dtype=float)
        self.rf_ = RandomForestRegressor(
            n_estimators=self.n_estimators, max_features=self.max_features,
            min_samples_leaf=self.min_samples_leaf, oob_score=True,
            n_jobs=-1, random_state=self.random_state).fit(X, y)
        oob = self.rf_.oob_prediction_
        Xs = X   # numeric columns are already standardised by the preprocessor
        if self.mode == "residual":
            target, Z = y - oob, Xs
        elif self.mode == "rf_feature":
            target, Z = y, np.column_stack([Xs, (oob - y.mean()) / y.std()])
            self.y_mean_, self.y_std_ = y.mean(), y.std()
        else:  # blend
            target, Z = y, Xs
        # n_ann > 1 -> a small bag of ANNs with different seeds (variance reduction)
        self.anns_ = [self._new_ann(self.random_state + i).fit(Z, target)
                      for i in range(self.n_ann)]
        return self

    def _ann_predict(self, X):
        Xs = X
        if self.mode == "rf_feature":
            rf_p = (self.rf_.predict(X) - self.y_mean_) / self.y_std_
            Xs = np.column_stack([Xs, rf_p])
        return np.mean([a.predict(Xs) for a in self.anns_], axis=0)

    def predict_rf(self, X):
        return self.rf_.predict(np.asarray(X))

    def predict_ann(self, X):
        X = np.asarray(X)
        p = self._ann_predict(X)
        return p

    def predict(self, X):
        X = np.asarray(X)
        rf_p = self.rf_.predict(X)
        ann_p = self._ann_predict(X)
        if self.mode == "residual":
            return rf_p + self.ann_weight * ann_p
        if self.mode == "rf_feature":
            return ann_p
        return self.rf_weight * rf_p + (1 - self.rf_weight) * ann_p

def make_hybrid_pipe(**kwargs):
    pipe = Pipeline([
        ("fe", FunctionTransformer(add_features)),
        ("pre", make_preprocessor(scale=True)),   # RF is invariant to scaling
        ("model", RFANNHybrid(**kwargs)),
    ])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p,
                                      inverse_func=np.expm1)


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------
def evaluate(y_true, y_pred):
    """Metrics on the original scale (LKR lakhs)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "MAPE_%": float(np.mean(np.abs(y_true - y_pred) / y_true) * 100),
        "RMSLE": float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))),
    }
