"""Esporta risultati REALI in file .tex: macro numeriche, tabelle, figure."""
import os
import re
import pandas as pd


def _letters(s: str) -> str:
    """I nomi di macro LaTeX ammettono solo lettere."""
    return re.sub(r"[^A-Za-z]", "", s.title())


def write_macros(results: pd.DataFrame, path="outputs/risultati_macro.tex"):
    """results: indice = (scenario, modello); colonne = metriche."""
    lines = ["% Generato automaticamente: NON modificare a mano\n"]
    for (scen, mod), row in results.iterrows():
        key = _letters(scen.split("(")[0]) + _letters(mod)
        lines.append(f"\\newcommand{{\\AUC{key}}}{{{row['ROC-AUC (OOF)']:.3f}}}")
        lines.append(f"\\newcommand{{\\AUC{key}Lo}}{{{row['IC95% basso']:.3f}}}")
        lines.append(f"\\newcommand{{\\AUC{key}Hi}}{{{row['IC95% alto']:.3f}}}")
        lines.append(f"\\newcommand{{\\PR{key}}}{{{row['PR-AUC (media fold)']:.3f}}}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def write_table(results: pd.DataFrame, path="outputs/tabella_risultati.tex"):
    t = results[["ROC-AUC (OOF)", "IC95% basso", "IC95% alto", "PR-AUC (media fold)", "Brier (OOF)"]].round(3)
    t.index.names = ["Scenario", "Modello"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t.to_latex(path, caption="Prestazioni in cross-validazione ripetuta (predizioni out-of-fold).",
               label="tab:risultati", escape=True, float_format="%.3f")
    return path
