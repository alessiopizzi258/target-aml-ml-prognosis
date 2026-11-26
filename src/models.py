"""Pipeline sklearn: imputazione e encoding DENTRO ogni fold (niente leakage)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False

MODEL_NAMES = ["Regressione Logistica", "Random Forest"] + (["XGBoost"] if HAS_XGB else [])


def prepare_X(df: pd.DataFrame, num, cat) -> pd.DataFrame:
    """Seleziona le colonne e uniforma i tipi (numeriche float, categoriche str/NaN)."""
    X = pd.DataFrame(index=df.index)
    for c in num:
        X[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    for c in cat:
        X[c] = df[c].map(lambda v: np.nan if pd.isna(v) else str(v)).astype(object)
    return X


def make_preprocessor(num, cat) -> ColumnTransformer:
    num_pipe = SimpleImputer(strategy="median", add_indicator=True)
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10, sparse_output=False)),
    ])
    transformers = []
    if num:
        transformers.append(("num", num_pipe, num))
    if cat:
        transformers.append(("cat", cat_pipe, cat))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


def make_model(name: str, seed: int = 42):
    if name == "Regressione Logistica":
        return Pipeline([("sc", StandardScaler()),
                         ("m", LogisticRegression(C=0.3, max_iter=5000, random_state=seed))])
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=5, max_features="sqrt",
                                      n_jobs=-1, random_state=seed)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1, random_state=seed)
    raise ValueError(name)


def build_pipeline(name: str, num, cat, seed: int = 42) -> Pipeline:
    return Pipeline([("pre", make_preprocessor(num, cat)), ("model", make_model(name, seed))])
