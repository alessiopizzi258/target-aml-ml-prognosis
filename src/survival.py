"""Survival analysis: Cox penalizzato con c-index in CV ripetuta."""
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from .models import make_preprocessor


def cv_cox(X, time, event, num, cat, n_splits=5, n_repeats=3, penalizer=0.1, seed=42):
    time, event = np.asarray(time), np.asarray(event)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    scores = []
    for tr, va in rskf.split(X, event):
        pre = make_preprocessor(num, cat)
        Xt = pre.fit_transform(X.iloc[tr])
        names = pre.get_feature_names_out()
        keep = Xt.std(axis=0) > 1e-8
        sc = StandardScaler().fit(Xt[:, keep])
        cols = names[keep]
        dtr = pd.DataFrame(sc.transform(Xt[:, keep]), columns=cols)
        dtr["T"], dtr["E"] = time[tr], event[tr]
        cph = CoxPHFitter(penalizer=penalizer).fit(dtr, "T", "E")
        dva = pd.DataFrame(sc.transform(pre.transform(X.iloc[va])[:, keep]), columns=cols)
        risk = cph.predict_partial_hazard(dva).values
        scores.append(concordance_index(time[va], -risk, event[va]))
    return np.array(scores)


def km_curve(time, event, label="Coorte"):
    k = KaplanMeierFitter()
    k.fit(time, event_observed=event, label=label)
    return k
