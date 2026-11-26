"""SHAP: da usare SOLO se il modello ha un potere predittivo non banale."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .models import build_pipeline


def shap_summary(X, y, num, cat, model_name="XGBoost", out_pdf="outputs/shap_summary.pdf", max_display=15):
    pipe = build_pipeline(model_name, num, cat).fit(X, y)
    Xt = pipe.named_steps["pre"].transform(X)
    names = list(pipe.named_steps["pre"].get_feature_names_out())
    Xt = pd.DataFrame(Xt, columns=names)
    model = pipe.named_steps["model"]
    try:
        sv = shap.TreeExplainer(model).shap_values(Xt)
        if isinstance(sv, list):          # RF: lista per classe
            sv = sv[1]
        elif getattr(sv, "ndim", 2) == 3:
            sv = sv[:, :, 1]
    except Exception:
        # versioni di SHAP che non leggono XGBoost recente: calcolo nativo (stessi valori degli alberi)
        if not hasattr(model, "get_booster"):
            raise
        import xgboost as xgb
        sv = model.get_booster().predict(xgb.DMatrix(np.asarray(Xt, dtype=float)), pred_contribs=True)[:, :-1]
    plt.figure(figsize=(9, 6))
    shap.summary_plot(sv, Xt, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(out_pdf)
    fig = plt.gcf()
    imp = pd.Series(np.abs(sv).mean(axis=0), index=names).sort_values(ascending=False)
    return fig, imp
