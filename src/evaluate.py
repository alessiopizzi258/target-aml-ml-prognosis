"""Valutazione: CV ripetuta, predizioni out-of-fold, intervalli di confidenza bootstrap."""
import numpy as np
from sklearn.base import clone
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold


def repeated_cv(pipe, X, y, n_splits=5, n_repeats=5, seed=42, progress=None):
    y = np.asarray(y)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    oof = np.zeros((n_repeats, len(y)))
    fold_auc, fold_pr = [], []
    total = n_splits * n_repeats
    for i, (tr, va) in enumerate(rskf.split(X, y)):
        m = clone(pipe).fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        oof[i // n_splits, va] = p
        fold_auc.append(roc_auc_score(y[va], p))
        fold_pr.append(average_precision_score(y[va], p))
        if progress:
            progress((i + 1) / total)
    return {"oof": oof.mean(axis=0), "fold_auc": np.array(fold_auc), "fold_pr": np.array(fold_pr)}


def bootstrap_ci(y, p, metric=roc_auc_score, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        vals.append(metric(y[idx], p[idx]))
    return np.percentile(vals, [2.5, 97.5])


def summarize(res, y):
    y = np.asarray(y)
    lo, hi = bootstrap_ci(y, res["oof"])
    return {
        "ROC-AUC (media fold)": res["fold_auc"].mean(),
        "ROC-AUC (dev.std fold)": res["fold_auc"].std(),
        "ROC-AUC (OOF)": roc_auc_score(y, res["oof"]),
        "IC95% basso": lo,
        "IC95% alto": hi,
        "PR-AUC (media fold)": res["fold_pr"].mean(),
        "Prevalenza (baseline PR-AUC)": y.mean(),
        "Brier (OOF)": brier_score_loss(y, res["oof"]),
    }
