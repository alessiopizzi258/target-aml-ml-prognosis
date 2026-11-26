"""
analisi_extra.py - Calcola le analisi AGGIUNTIVE per la tesi e scrive figure, tabelle e paragrafi LaTeX
con i numeri veri (sempre come macro). Si lancia nella cartella tesi_aml, dopo aver messo i file clinici in
data/clinical_supplements.

    python analisi_extra.py                       # tutte le sezioni
    python analisi_extra.py --solo curve pesi     # solo alcune
    python analisi_extra.py --ripetizioni 2 --iter 4    # prova veloce

Sezioni: flusso, variabili, curve, pesi, tuning, sopravvivenza, shap, landmark
Output (cartella outputs/extra/):
    extra_3_3.tex  (selezione pazienti, distribuzioni, mortalita' per categoria)   -> paragrafo 3.3
    extra_3_4.tex  (elenco delle variabili)                                        -> paragrafo 3.4
    extra_4_2.tex  (scelta degli iperparametri con validazione annidata)           -> paragrafo 4.2
    extra_4_3.tex  (confronto con i pesi di classe)                                -> paragrafo 4.3
    extra_5_1.tex  (curve ROC/PR, calibrazione, soglia, F1)                        -> paragrafo 5.1
    extra_5_2.tex  (importanza per variabile, SHAP post-induzione)                 -> paragrafo 5.2
    extra_5_4.tex  (Cox post-induzione, analisi di sensibilita' sull'OS)           -> paragrafo 5.4
    risultati_extra_macro.tex, figure *.pdf
Poi, nella cartella della tesi:  python inserisci_extra.py <percorso>/outputs/extra

LIMITI: i testi descrivono SOLO i numeri calcolati (frasi condizionate ai valori). Le interpretazioni
(perche' un risultato e' cosi') restano tue: aggiungile sotto ogni sottosezione.
"""
import argparse
import json
import os
import re
import time
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, matthews_corrcoef,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import ID, binary_target, clean, load_raw, merge_patients, survival_target, valid_os
from src.evaluate import repeated_cv
from src.features import DIAG_CAT, DIAG_NUM, POST_CAT, POST_NUM, SCENARIOS, get_columns
from src.models import MODEL_NAMES, build_pipeline, make_preprocessor, prepare_X
from src.survival import cv_cox, km_curve

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

SEED = 42
SC = {"Base": "Solo età e sesso (baseline)", "Diag": "Diagnosi (tutte le feature cliniche)",
      "Post": "Post-induzione (+ MRD, CR1, risk group)"}
SC = {k: next(s for s in SCENARIOS if s.startswith(v.split(" (")[0])) for k, v in SC.items()}
ABBR = {"Regressione Logistica": "LR", "Random Forest": "RF", "XGBoost": "XGBoost"}
NOMI_SC = {"Base": "età e sesso", "Diag": "alla diagnosi", "Post": "post-induzione"}
DESCR = {
    "Age_years": "Età alla diagnosi in anni", "WBC at Diagnosis": "Conta dei globuli bianchi alla diagnosi ($\\times10^3/\\mu$L)",
    "Bone marrow leukemic blast percentage (%)": "Percentuale di blasti nel midollo osseo",
    "Peripheral blasts (%)": "Percentuale di blasti nel sangue periferico", "Gender": "Sesso",
    "Race": "Razza dichiarata", "Ethnicity": "Etnia dichiarata", "CNS disease": "Coinvolgimento del sistema nervoso centrale",
    "Chloroma": "Presenza di cloroma (massa extramidollare)", "FAB Category": "Classificazione morfologica FAB",
    "MLL": "Riarrangiamento di KMT2A (MLL)", "Primary Cytogenetic Code": "Categoria citogenetica primaria",
    "Cytogenetic Complexity": "Complessità del cariotipo", "FLT3/ITD positive?": "Duplicazione interna in tandem di FLT3",
    "FLT3 PM": "Mutazione puntiforme di FLT3", "NPM mutation": "Mutazione di NPM1", "CEBPA mutation": "Mutazione di CEBPA",
    "WT1 mutation": "Mutazione di WT1", "c-Kit Mutation Exon 8": "Mutazione di c-Kit, esone 8",
    "c-Kit Mutation Exon 17": "Mutazione di c-Kit, esone 17", "MRD % at end of course 1": "Malattia minima residua (\\%) a fine corso 1",
    "MRD at end of course 1": "Malattia minima residua (positiva/negativa) a fine corso 1",
    "CR status at end of course 1": "Stato di remissione a fine corso 1", "Risk group": "Gruppo di rischio riportato nei dati",
}
LABEL = {"Age_years": "età alla diagnosi", "WBC at Diagnosis": "WBC alla diagnosi", "inv(16)": "inv(16)", "t(8;21)": "t(8;21)",
         "CEBPA mutation": "mutazione di CEBPA", "NPM mutation": "mutazione di NPM1", "FLT3/ITD positive?": "FLT3-ITD",
         "Risk group": "gruppo di rischio", "MRD at end of course 1": "MRD a fine corso 1 (positiva/negativa)",
         "MRD % at end of course 1": "MRD (\\%) a fine corso 1", "CR status at end of course 1": "remissione a fine corso 1",
         "Peripheral blasts (%)": "blasti periferici", "Bone marrow leukemic blast percentage (%)": "blasti nel midollo osseo",
         "Primary Cytogenetic Code": "codice citogenetico primario", "FAB Category": "classificazione FAB", "MLL": "riarrangiamento di KMT2A",
         "WT1 mutation": "mutazione di WT1", "Gender": "sesso", "Cytogenetic Complexity": "complessità del cariotipo"}
OUT = "outputs/extra"
MAC = {}
A = None
D = {}


def esc(s):
    return (str(s).replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
            .replace("#", "\\#"))


def m(chiave, valore, fmt="{:.3f}"):
    """Registra una macro (nome con sole lettere) e restituisce il comando da usare nel testo."""
    nome = "Ex" + re.sub(r"[^A-Za-z]", "", chiave)
    MAC[nome] = valore if isinstance(valore, str) else fmt.format(valore)
    return "\\" + nome + "{}"


def salva_macro():
    p = os.path.join(OUT, "_macro.json")
    vecchie = json.load(open(p)) if os.path.exists(p) else {}
    vecchie.update(MAC)
    json.dump(vecchie, open(p, "w"), indent=1)
    with open(os.path.join(OUT, "risultati_extra_macro.tex"), "w", encoding="utf-8") as f:
        f.write("% Generato da analisi_extra.py: NON modificare a mano\n")
        for k, v in sorted(vecchie.items()):
            f.write("\\newcommand{\\%s}{%s}\n" % (k, v))


def scrivi(nome, testo):
    with open(os.path.join(OUT, nome), "w", encoding="utf-8") as f:
        f.write(testo)
    print(f"   scritto {nome}")


def lab(v):
    return LABEL.get(v, esc(v))


def passa(a, b, fmt="{:.3f}", unita=""):
    fa, fb = fmt.format(a), fmt.format(b)
    return f"resta pari a {fa}{unita}" if fa == fb else f"passa da {fa}{unita} a {fb}{unita}"


def confronto(delta, tol=0.01):
    return "praticamente invariata" if abs(delta) <= tol else ("più alta" if delta > 0 else "più bassa")


def carica():
    raw = load_raw(A.dati)
    pat, _ = merge_patients(raw)
    d = clean(pat)
    sub, y = binary_target(d, 1095)
    D.update(raw=raw, pat=pat, d=d, sub=sub, y=y)


def tabella(righe, intest, colspec, caption, label, piccola=True):
    s = ["\\begin{table}[htbp]\n\\centering" + ("\\small" if piccola else ""), f"\\caption{{{caption}}}\\label{{{label}}}",
         "\\resizebox{\\textwidth}{!}{%", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", " & ".join(intest) + " \\\\", "\\midrule"]
    s += [" & ".join(r) + " \\\\" for r in righe]
    s += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    return "\n".join(s) + "\n"


def figura(file, caption, label, larg="0.95"):
    return (f"\\begin{{figure}}[htbp]\n\\centering\n\\includegraphics[width={larg}\\textwidth]{{extra/{file}}}\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{figure}}\n")


# ---------------------------------------------------------------- flusso + distribuzioni
def sez_flusso():
    raw, pat, d, sub, y = (D[k] for k in ("raw", "pat", "d", "sub", "y"))
    dv = valid_os(d)
    ncens = len(dv) - len(sub)
    morti_os = int((dv["Vital Status"] == "Dead").sum())
    righe = [["Righe nei cinque file clinici", f"{len(raw)}"], ["Pazienti unici dopo l'unione dei file", f"{len(pat)}"],
             ["Pazienti con stato vitale e tempo di OS noti", f"{len(dv)}"],
             ["di cui deceduti (a qualunque tempo)", f"{morti_os}"],
             ["Esclusi dal target binario (censurati prima di 1095 giorni)", f"{ncens}"],
             ["Pazienti nel target binario a 3 anni", f"{len(sub)}"], ["di cui deceduti entro 3 anni", f"{int(y.sum())}"]]
    t = ["\\subsection{Selezione dei pazienti}",
         f"La Tabella~\\ref{{tab:flusso}} riassume i passaggi che portano dai file originali alle coorti analizzate. "
         f"Dei {m('NPazOS', len(dv), '{}')} pazienti con stato vitale e tempo di OS noti, {m('NMortiOS', morti_os, '{}')} sono deceduti "
         f"e {m('NCens', ncens, '{}')} sono censurati prima di 1095 giorni: questi ultimi sono esclusi dal target binario, che comprende "
         f"\\NBinario{{}} pazienti (\\NMorti{{}} decessi entro 3 anni, pari al \\PercMorti\\%). "
         "Il modello di sopravvivenza non richiede questa esclusione e usa tutti i pazienti con tempo di OS positivo.\n",
         tabella(righe, ["Passaggio", "Pazienti"], "lr", "Dai file clinici alle coorti di analisi.", "tab:flusso", False)]
    # distribuzioni
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for lab, col in ((0, "#2c7fb8"), (1, "#c0392b")):
        s = sub[y == lab]
        ax[0].hist(s["Age_years"].dropna(), bins=20, alpha=0.55, color=col, density=True,
                   label=("Sopravvissuti" if lab == 0 else "Deceduti entro 3 anni"))
        w = s["WBC at Diagnosis"].dropna()
        ax[1].hist(np.log10(w[w > 0]), bins=20, alpha=0.55, color=col, density=True)
    ax[0].set_xlabel("Età alla diagnosi (anni)"); ax[0].set_ylabel("Densità"); ax[0].legend()
    ax[1].set_xlabel("WBC alla diagnosi (scala logaritmica, $\\log_{10}$)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "distribuzioni_eta_wbc.pdf")); plt.close()
    t.append("\\subsection{Distribuzione e mortalità per variabile}")
    t.append("La Figura~\\ref{fig:distribuzioni} confronta la distribuzione di età e conta dei globuli bianchi alla diagnosi tra i pazienti "
             "sopravvissuti e quelli deceduti entro 3 anni.\n")
    t.append(figura("distribuzioni_eta_wbc.pdf", "Distribuzione dell'età e del WBC alla diagnosi per esito a 3 anni.", "fig:distribuzioni"))
    # mortalita' per categoria
    righe, frasi = [], []
    ETICHETTE = {"t(8;21)": ("t(8;21)", "T821"), "inv(16)": ("inv(16)", "Inv16"), "MLL": ("riarrangiamento di KMT2A", "Kmt"),
                 "FLT3/ITD positive?": ("FLT3-ITD", "Flt"), "NPM mutation": ("mutazione di NPM1", "Npm"),
                 "CEBPA mutation": ("mutazione di CEBPA", "Cebpa"), "WT1 mutation": ("mutazione di WT1", "Wt")}
    for v in ["Gender", "FAB Category", "CNS disease", "t(8;21)", "inv(16)", "MLL", "FLT3/ITD positive?", "NPM mutation",
              "CEBPA mutation", "WT1 mutation"]:
        if v not in sub.columns:
            continue
        g = pd.DataFrame({"c": sub[v].astype("string").fillna("Missing"), "y": y}).groupby("c")["y"].agg(["size", "mean"])
        for cat, r in g.iterrows():
            if r["size"] >= 20:
                righe.append([esc(v), esc(cat), f"{int(r['size'])}", f"{100 * r['mean']:.1f}"])
        if v in ETICHETTE and {"Yes", "No"} <= set(g.index) and g.loc["Yes", "size"] >= 20:
            lab, k = ETICHETTE[v]
            frasi.append(f"{lab}: {m('Mort' + k + 'Si', 100 * g.loc['Yes', 'mean'], '{:.1f}')}\\% con l'alterazione, "
                         f"{m('Mort' + k + 'No', 100 * g.loc['No', 'mean'], '{:.1f}')}\\% senza")
    for v, lab in (("Age_years", "Età (quartili)"), ("WBC at Diagnosis", "WBC (quartili)")):
        q = pd.qcut(sub[v], 4, duplicates="drop")
        g = pd.DataFrame({"c": q, "y": y}).groupby("c", observed=True)["y"].agg(["size", "mean"])
        for iv, r in g.iterrows():
            righe.append([lab, f"{iv.left:.1f}--{iv.right:.1f}", f"{int(r['size'])}", f"{100 * r['mean']:.1f}"])
    unk = []
    for v in ["FLT3/ITD positive?", "NPM mutation", "CEBPA mutation", "WT1 mutation"]:
        if v in sub.columns:
            u = sub[v].astype("string").eq("Unknown").fillna(False).to_numpy()
            if u.sum() >= 20:
                unk.append((int(u.sum()), 100 * float(y[u].mean()), float(sub.loc[u, "Overall Survival Time in Days"].median())))
    frase_unk = ""
    if unk:
        frase_unk = ("Per FLT3-ITD, NPM1, CEBPA e WT1 i pazienti con stato ``Unknown'' (tra "
                     f"{m('UnkNMin', min(x[0] for x in unk), '{}')} e {m('UnkNMax', max(x[0] for x in unk), '{}')} per variabile) hanno una mortalità a 3 anni "
                     f"compresa tra {m('UnkMortMin', min(x[1] for x in unk), '{:.1f}')}\\% e {m('UnkMortMax', max(x[1] for x in unk), '{:.1f}')}\\%, "
                     f"molto inferiore al \\PercMorti\\% complessivo, e un tempo di OS mediano compreso tra {m('UnkOSMin', min(x[2] for x in unk), '{:.0f}')} e "
                     f"{m('UnkOSMax', max(x[2] for x in unk), '{:.0f}')} giorni, contro {m('OSMedTutti', float(sub['Overall Survival Time in Days'].median()), '{:.0f}')} "
                     "giorni nella coorte del target binario. L'assenza di questa informazione non sembra quindi casuale: potrebbe riflettere la coorte di provenienza "
                     "e la lunghezza del follow-up, e un modello potrebbe sfruttarla come indicatore indiretto. È un limite dello studio (capitolo~6).")
    t.append("La Tabella~\\ref{tab:mortcat} riporta, per alcune variabili, la percentuale di decessi entro 3 anni in ciascuna categoria "
             "(categorie con almeno 20 pazienti). Si tratta di frequenze osservate, senza correzione per le altre variabili: "
             "non indicano un effetto indipendente di ciascun fattore.")
    if frasi:
        t.append("Per le principali alterazioni la percentuale di decessi entro 3 anni è la seguente: " + "; ".join(frasi) + ".")
    if frase_unk:
        t.append(frase_unk)
    t.append("")
    t.append(tabella(righe, ["Variabile", "Categoria", "Pazienti", "Decessi a 3 anni (\\%)"], "llrr",
                     "Mortalità a 3 anni per categoria di alcune variabili.", "tab:mortcat"))
    scrivi("extra_3_3.tex", "\n".join(t))


# ---------------------------------------------------------------- elenco variabili
def sez_variabili():
    dv = valid_os(D["d"])
    elenco = [(v, "num", "D") for v in DIAG_NUM] + [(v, "cat", "D") for v in DIAG_CAT] + \
             [(v, "num", "P") for v in POST_NUM] + [(v, "cat", "P") for v in POST_CAT]
    righe = []
    for v, tipo, sc in elenco:
        if v not in dv.columns:
            continue
        s = dv[v]
        mis = 100 * s.isna().mean()
        if tipo == "num":
            x = pd.to_numeric(s, errors="coerce").dropna()
            sint = f"mediana {x.median():.1f} (IQR {x.quantile(.25):.1f}--{x.quantile(.75):.1f})" if len(x) else "--"
        else:
            vc = s.dropna().astype(str).value_counts(normalize=True)
            sint = "; ".join(f"{esc(k)} {100 * p:.0f}\\%" for k, p in vc.head(3).items()) + (f" ({len(vc)} categorie)" if len(vc) > 3 else "")
        righe.append([esc(v), DESCR.get(v, "--"), "diagnosi" if sc == "D" else "post-ind.", f"{mis:.0f}", sint])
    corpo = "\n".join(" & ".join(r) + " \\\\\n\\hline" for r in righe)
    t = ["\\subsection{Elenco delle variabili utilizzate}",
         f"La Tabella~\\ref{{tab:variabili}} elenca le {len(righe)} variabili presenti nel supplemento clinico e usate nei tre scenari: "
         "per ciascuna sono indicati il significato, il momento in cui l'informazione è disponibile, la percentuale di valori mancanti e una "
         f"sintesi della distribuzione sui {m('NVarPaz', len(dv), '{}')} pazienti con OS noto. Le variabili ``post-induzione'' compaiono solo "
         "nel terzo scenario.\n",
         "{\\small\\setlength{\\tabcolsep}{3pt}\n\\begin{longtable}{p{3.2cm}p{3.8cm}p{1.9cm}p{1.2cm}p{3.7cm}}",
         "\\caption{Variabili utilizzate nei modelli.}\\label{tab:variabili}\\\\\n\\hline\nVariabile & Significato & Disponibile & Mancanti (\\%) & Sintesi \\\\\n\\hline\n\\endfirsthead",
         "\\hline\nVariabile & Significato & Disponibile & Mancanti (\\%) & Sintesi \\\\\n\\hline\n\\endhead", corpo,
         "\\end{longtable}\n}\n"]
    scrivi("extra_3_4.tex", "\n".join(t))


# ---------------------------------------------------------------- curve, calibrazione, soglie
def met(y, p, t):
    pr = (p >= t).astype(int)
    return (precision_score(y, pr, zero_division=0), recall_score(y, pr, zero_division=0),
            f1_score(y, pr, zero_division=0), matthews_corrcoef(y, pr))


def sez_curve():
    sub, y = D["sub"], D["y"]
    prev = y.mean()
    oof = {}
    for sk, sc in SC.items():
        num, cat = get_columns(sc, sub.columns)
        X = prepare_X(sub, num, cat)
        for mn in MODEL_NAMES:
            print(f"   {sk} / {mn}")
            oof[(sk, mn)] = repeated_cv(build_pipeline(mn, num, cat, SEED), X, y, n_repeats=A.ripetizioni, seed=SEED)["oof"]
    for nome, fun, xl, yl in (("roc", "roc", "Tasso di falsi positivi", "Sensibilità"), ("pr", "pr", "Sensibilità (recall)", "Precisione")):
        fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
        for a_, (sk, sc) in zip(ax, SC.items()):
            for mn in MODEL_NAMES:
                p = oof[(sk, mn)]
                if fun == "roc":
                    f, tp, _ = roc_curve(y, p)
                    a_.plot(f, tp, label=f"{ABBR[mn]} (AUC {roc_auc_score(y, p):.3f})")
                else:
                    pr_, rc, _ = precision_recall_curve(y, p)
                    a_.plot(rc, pr_, label=f"{ABBR[mn]} (AP {average_precision_score(y, p):.3f})")
            if fun == "roc":
                a_.plot([0, 1], [0, 1], "k--", lw=0.8)
            else:
                a_.axhline(prev, color="k", ls="--", lw=0.8)
            a_.set_title(f"Scenario: {NOMI_SC[sk]}"); a_.set_xlabel(xl); a_.set_ylabel(yl); a_.legend(loc="best", fontsize=8)
        plt.tight_layout(); plt.savefig(os.path.join(OUT, f"{nome}_scenari.pdf")); plt.close()
    fig, ax = plt.subplots(figsize=(5.2, 4.6)); ece = {}
    for sk in SC:
        p = oof[(sk, "XGBoost")]
        fy, fx = calibration_curve(y, p, n_bins=8, strategy="quantile")
        ax.plot(fx, fy, "o-", label=NOMI_SC[sk])
        q = pd.qcut(p, 8, duplicates="drop")
        gg = pd.DataFrame({"p": p, "y": y, "q": q}).groupby("q", observed=True).agg(n=("y", "size"), p=("p", "mean"), y=("y", "mean"))
        ece[sk] = float((gg["n"] * (gg["p"] - gg["y"]).abs()).sum() / gg["n"].sum())
    ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.set_xlabel("Probabilità stimata (media per ottavo)"); ax.set_ylabel("Frequenza osservata di decesso")
    ax.legend(); plt.tight_layout(); plt.savefig(os.path.join(OUT, "calibrazione_xgboost.pdf")); plt.close()
    righe, ris = [], {}
    for sk in ("Diag", "Post"):
        for mn in MODEL_NAMES:
            for tn, t in (("0.5", 0.5), (f"{100 * prev:.1f}\\% (prevalenza)", prev)):
                pr_, rc, f1, mc = met(y, oof[(sk, mn)], t)
                ris[(sk, mn, "mezzo" if t == 0.5 else "prev")] = (pr_, rc, f1, mc)
                righe.append([NOMI_SC[sk], ABBR[mn], tn, f"{100 * pr_:.1f}", f"{100 * rc:.1f}", f"{f1:.3f}", f"{mc:.3f}"])
    X1 = ris[("Diag", "XGBoost", "mezzo")]; X2 = ris[("Diag", "XGBoost", "prev")]
    Y2 = ris[("Post", "XGBoost", "prev")]
    spread = max(max(roc_auc_score(y, oof[(sk, mn)]) for mn in MODEL_NAMES) - min(roc_auc_score(y, oof[(sk, mn)]) for mn in MODEL_NAMES)
                 for sk in ("Diag", "Post"))
    frase_curve = ("Nei due scenari con più variabili le curve dei tre modelli sono in larga parte sovrapposte, coerentemente con AUC che differiscono "
                   f"al massimo di {m('CurveSpread', spread)}." if spread <= 0.02 else
                   f"Nei due scenari con più variabili le curve dei modelli si distinguono (differenza massima di ROC-AUC {m('CurveSpread', spread)}).")
    t = ["\\subsection{Curve ROC e precisione-sensibilità}",
         "La Figura~\\ref{fig:roc} mostra le curve ROC dei tre modelli nei tre scenari, calcolate sulle predizioni out-of-fold; la "
         "Figura~\\ref{fig:pr} mostra le curve di precisione e sensibilità, con la prevalenza dei decessi come riferimento di un classificatore casuale. "
         + frase_curve + "\n",
         figura("roc_scenari.pdf", "Curve ROC (predizioni out-of-fold) nei tre scenari.", "fig:roc"),
         figura("pr_scenari.pdf", "Curve precisione-sensibilità nei tre scenari; la linea tratteggiata è la prevalenza.", "fig:pr"),
         "\\subsection{Calibrazione}",
         "La Figura~\\ref{fig:calibrazione} confronta, per XGBoost, la probabilità stimata con la frequenza di decesso osservata, dopo aver "
         "raggruppato i pazienti in ottavi di probabilità stimata. Come misura sintetica si usa la differenza assoluta media, pesata per numero di "
         f"pazienti, tra le due quantità: vale {m('EceBase', ece['Base'])} con la sola età e sesso, {m('EceDiag', ece['Diag'])} alla diagnosi e "
         f"{m('EcePost', ece['Post'])} dopo l'induzione (valori più bassi indicano una migliore calibrazione).\n",
         figura("calibrazione_xgboost.pdf", "Calibrazione di XGBoost nei tre scenari (predizioni out-of-fold).", "fig:calibrazione", "0.6"),
         "\\subsection{Soglia di classificazione, precisione, sensibilità e F1}",
         "Le metriche precedenti valutano le probabilità; per ottenere una classificazione occorre fissare una soglia. La Tabella~\\ref{tab:soglie} "
         "riporta precisione, sensibilità, F1-score e coefficiente di correlazione di Matthews (MCC) con due soglie non ottimizzate sui dati di "
         "valutazione: 0.5 e la prevalenza dei decessi.",
         f"Con la soglia 0.5, XGBoost nello scenario alla diagnosi ha una sensibilità del {m('SensDiagMezzo', 100 * X1[1], '{:.1f}')}\\% e una "
         f"precisione del {m('PrecDiagMezzo', 100 * X1[0], '{:.1f}')}\\%. Con la soglia pari alla prevalenza la sensibilità diventa "
         f"{m('SensDiagPrev', 100 * X2[1], '{:.1f}')}\\%, la precisione {m('PrecDiagPrev', 100 * X2[0], '{:.1f}')}\\% e l'F1-score "
         f"{m('FDiagPrev', X2[2])}; dopo l'induzione l'F1-score alla stessa soglia è {m('FPostPrev', Y2[2])}. "
         "L'F1-score dipende dalla soglia scelta, mentre la ROC-AUC no.\n",
         tabella(righe, ["Scenario", "Modello", "Soglia", "Precisione (\\%)", "Sensibilità (\\%)", "F1", "MCC"], "llcrrrr",
                 "Metriche di classificazione a due soglie (predizioni out-of-fold).", "tab:soglie")]
    scrivi("extra_5_1.tex", "\n".join(t))


# ---------------------------------------------------------------- pesi di classe
def pipe_pesato(nome, num, cat, y):
    if nome == "Regressione Logistica":
        mod = Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(C=0.3, max_iter=5000, class_weight="balanced", random_state=SEED))])
    elif nome == "Random Forest":
        mod = RandomForestClassifier(n_estimators=500, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=SEED)
    else:
        mod = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                            n_jobs=-1, random_state=SEED, scale_pos_weight=float((y == 0).sum() / (y == 1).sum()))
    return Pipeline([("pre", make_preprocessor(num, cat)), ("model", mod)])


def sez_pesi():
    sub, y = D["sub"], D["y"]
    righe, dauc, bn, bs, rn, rs_, pn, ps = [], [], [], [], [], [], [], []
    for sk in ("Diag", "Post"):
        num, cat = get_columns(SC[sk], sub.columns)
        X = prepare_X(sub, num, cat)
        for mn in MODEL_NAMES:
            print(f"   {sk} / {mn}")
            ris = {}
            for peso in (False, True):
                pipe = pipe_pesato(mn, num, cat, y) if peso else build_pipeline(mn, num, cat, SEED)
                r = repeated_cv(pipe, X, y, n_repeats=A.ripetizioni, seed=SEED)
                p = r["oof"]; pr0, rc, f1, _ = met(y, p, 0.5)
                ris[peso] = (roc_auc_score(y, p), r["fold_pr"].mean(), brier_score_loss(y, p), f1, rc)
                righe.append([NOMI_SC[sk], ABBR[mn], "sì" if peso else "no", f"{ris[peso][0]:.3f}", f"{ris[peso][1]:.3f}",
                              f"{ris[peso][2]:.3f}", f"{ris[peso][3]:.3f}", f"{100 * ris[peso][4]:.1f}"])
            dauc.append(ris[True][0] - ris[False][0]); bn.append(ris[False][2]); bs.append(ris[True][2])
            rn.append(ris[False][4]); rs_.append(ris[True][4])
    peggio = np.mean(bs) > np.mean(bn) + 0.005
    t = ["\\subsection{Confronto con i pesi di classe}",
         "Come verifica, i tre modelli sono stati riaddestrati con pesi di classe (pesi bilanciati per la regressione logistica e il random forest; "
         "peso della classe positiva pari al rapporto tra negativi e positivi per XGBoost), mantenendo identiche validazione e variabili. "
         "La Tabella~\\ref{tab:pesi} riporta i risultati. La ROC-AUC è "
         f"{confronto(float(np.mean(dauc)), 0.01)} con i pesi (variazione assoluta massima {m('PesiDAucMax', float(np.max(np.abs(dauc))))}), "
         f"mentre il punteggio di Brier medio passa da {m('PesiBrierNo', float(np.mean(bn)))} a {m('PesiBrierSi', float(np.mean(bs)))}"
         + (": le probabilità stimate risultano meno aderenti alle frequenze osservate, perché i pesi spostano le probabilità verso l'alto. "
            if peggio else ", senza un cambiamento rilevante. ")
         + (f"Con la soglia 0.5 la sensibilità media passa dal {m('PesiSensNo', 100 * np.mean(rn), '{:.1f}')}\\% al {m('PesiSensSi', 100 * np.mean(rs_), '{:.1f}')}\\%. "
            if np.mean(rs_) > np.mean(rn) + 0.01 else "Con la soglia 0.5 la sensibilità non aumenta in modo rilevante. ")
         + "Non è stata applicata alcuna tecnica di sovracampionamento come SMOTE.\n",
         tabella(righe, ["Scenario", "Modello", "Pesi", "ROC-AUC", "PR-AUC", "Brier", "F1 (0.5)", "Sensibilità (0.5, \\%)"], "llcrrrrr",
                 "Confronto con e senza pesi di classe (predizioni out-of-fold).", "tab:pesi")]
    scrivi("extra_4_3.tex", "\n".join(t))


# ---------------------------------------------------------------- tuning annidato
SPAZI = {
    "XGBoost": {"model__max_depth": [2, 3, 4, 5], "model__learning_rate": [0.02, 0.05, 0.1], "model__n_estimators": [100, 200, 300, 500],
                "model__subsample": [0.6, 0.8, 1.0], "model__colsample_bytree": [0.6, 0.8, 1.0]},
    "Random Forest": {"model__n_estimators": [200, 500], "model__min_samples_leaf": [1, 3, 5, 10, 20], "model__max_features": ["sqrt", 0.3, 0.5]},
}


def sez_tuning():
    sub, y = D["sub"], D["y"]
    righe, scelte, dd = [], [], []
    for sk in ("Diag", "Post"):
        num, cat = get_columns(SC[sk], sub.columns)
        X = prepare_X(sub, num, cat)
        for mn in ("Random Forest", "XGBoost"):
            if mn not in MODEL_NAMES:
                continue
            print(f"   {sk} / {mn}")
            pf, pt = np.zeros(len(y)), np.zeros(len(y)); best = []
            for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
                base = build_pipeline(mn, num, cat, SEED)
                pf[va] = base.fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
                rs = RandomizedSearchCV(build_pipeline(mn, num, cat, SEED), SPAZI[mn], n_iter=A.iter, cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
                                        scoring="roc_auc", random_state=SEED, n_jobs=1)
                rs.fit(X.iloc[tr], y[tr]); pt[va] = rs.predict_proba(X.iloc[va])[:, 1]; best.append(rs.best_params_)
            a_f, a_t = roc_auc_score(y, pf), roc_auc_score(y, pt); dd.append(a_t - a_f)
            righe.append([NOMI_SC[sk], ABBR[mn], f"{a_f:.3f}", f"{a_t:.3f}", f"{a_t - a_f:+.3f}"])
            for k in SPAZI[mn]:
                vals = pd.Series([str(b[k]) for b in best]).value_counts()
                scelte.append([NOMI_SC[sk], ABBR[mn], esc(k.replace("model__", "")), esc(", ".join(f"{a} ({b}/5)" for a, b in vals.items()))])
    ddm = float(np.mean(dd))
    t = ["\\subsection{Scelta degli iperparametri con validazione annidata}",
         "Gli iperparametri dei modelli principali sono stati fissati a priori. Per valutare quanto questa scelta incida, per il random forest e per XGBoost "
         f"è stata eseguita una validazione annidata: in ciascuno dei 5 fold esterni gli iperparametri sono scelti con una ricerca casuale di {m('TuningIter', A.iter, '{}')} "
         "combinazioni valutate con una validazione incrociata interna a 3 fold (criterio: ROC-AUC), e il modello scelto è poi valutato sul fold esterno, "
         "che non è mai usato per la scelta. Gli spazi di ricerca sono: " + "; ".join(
             f"per {ABBR[k]}, " + ", ".join(f"{esc(p.replace('model__', ''))} in {{{', '.join(str(x) for x in v)}}}" for p, v in SPAZI[k].items())
             for k in ("Random Forest", "XGBoost")) + ".\n",
         "La Tabella~\\ref{tab:tuning} confronta la ROC-AUC con iperparametri fissi e con scelta annidata sugli stessi fold. In media la ROC-AUC con la "
         f"scelta annidata è {confronto(ddm, 0.01)} (differenza media {m('TuningDMedia', ddm, '{:+.3f}')}); i valori scelti nei fold sono riportati nella Tabella~\\ref{{tab:tuning2}}. "
         "Il confronto usa una sola ripetizione della validazione esterna ed è quindi più rumoroso di quello dei risultati principali.\n",
         tabella(righe, ["Scenario", "Modello", "ROC-AUC fissi", "ROC-AUC annidata", "Differenza"], "llrrr",
                 "ROC-AUC con iperparametri fissi e con scelta annidata (5 fold esterni).", "tab:tuning", False),
         tabella(scelte, ["Scenario", "Modello", "Iperparametro", "Valori scelti (fold)"], "lllp{6cm}",
                 "Valori degli iperparametri scelti nei 5 fold esterni.", "tab:tuning2")]
    scrivi("extra_4_2.tex", "\n".join(t))


# ---------------------------------------------------------------- Cox post-induzione e sensibilita' OS
def sez_sopravvivenza():
    raw, d, y_main = D["raw"], D["d"], D["y"]
    c = "Overall Survival Time in Days"
    r = raw[[ID, c, "Vital Status"]].copy(); r[c] = pd.to_numeric(r[c], errors="coerce")
    r["Vital Status"] = r["Vital Status"].astype("string").str.strip().str.capitalize()
    mx = r.groupby(ID)[c].max(); dead = r.groupby(ID)["Vital Status"].apply(lambda s: bool((s == "Dead").any()))
    d2 = d.copy(); d2[c] = d2[ID].map(mx)
    d2["Vital Status"] = d2["Vital Status"].where(d2["Vital Status"].isna(), np.where(d2[ID].map(dead).astype(bool), "Dead", "Alive"))
    num, cat = get_columns(SC["Diag"], d.columns)
    righe, res = [], {}
    for nome, dd in (("primo valore non nullo (analisi principale)", d), ("valore massimo tra i file", d2)):
        print("   OS:", nome)
        ds, T, E = survival_target(dd); k = km_curve(T, E)
        s3 = float(k.predict(1095))
        sb, yb = binary_target(dd, 1095)
        auc = roc_auc_score(yb, repeated_cv(build_pipeline("XGBoost", num, cat, SEED), prepare_X(sb, num, cat), yb, n_repeats=A.ripetizioni, seed=SEED)["oof"])
        ci = cv_cox(prepare_X(ds, num, cat), T, E, num, cat, n_repeats=A.ripetizioni, seed=SEED)
        res[nome] = (s3, auc, ci.mean())
        righe.append([nome, f"{len(ds)}", f"{int(E.sum())}", f"{100 * s3:.1f}", f"{len(sb)}", f"{auc:.3f}", f"{ci.mean():.3f}"])
    a, b = res["primo valore non nullo (analisi principale)"], res["valore massimo tra i file"]
    print("   Cox post-induzione")
    dsP, TP, EP = survival_target(d); numP, catP = get_columns(SC["Post"], d.columns)
    cp = cv_cox(prepare_X(dsP, numP, catP), TP, EP, numP, catP, n_repeats=A.ripetizioni, seed=SEED)
    t = ["\\subsection{Modello di Cox nello scenario post-induzione}",
         f"Con le stesse variabili dello scenario post-induzione il modello di Cox ottiene un indice di concordanza di {m('CindexPost', cp.mean())} "
         f"(deviazione standard {m('CindexPostSD', cp.std())}) su {m('NCoxPost', len(dsP), '{}')} pazienti, da confrontare con il valore ottenuto con le sole "
         "variabili alla diagnosi riportato nel paragrafo precedente.\n",
         "\\subsection{Analisi di sensibilità sul tempo di sopravvivenza}",
         "Per \\NConflittiOS{} pazienti il tempo di OS è discordante tra i file. Come controllo di robustezza, le analisi sono state ripetute "
         "usando, per ciascun paziente, il valore massimo tra i file invece del primo non nullo (un paziente è considerato deceduto se risulta tale in "
         f"almeno un file). La Tabella~\\ref{{tab:sensibilita}} riporta i risultati. La sopravvivenza a 3 anni stimata {passa(100 * a[0], 100 * b[0], '{:.1f}', '\\%')}, "
         f"la ROC-AUC di XGBoost alla diagnosi {passa(a[1], b[1])} e il c-index del modello di Cox {passa(a[2], b[2])}. "
         + ("Le differenze sono trascurabili: le conclusioni non dipendono dalla scelta della definizione del tempo di OS.\n" if
            max(abs(b[0] - a[0]), abs(b[1] - a[1]), abs(b[2] - a[2])) <= 0.01 else
            "Almeno una delle quantità cambia in modo apprezzabile: la scelta della definizione del tempo di OS va considerata tra i limiti dello studio.\n"),
         tabella(righe, ["Definizione di OS", "Pazienti", "Decessi", "S(3 anni) (\\%)", "N binario", "ROC-AUC XGBoost", "c-index Cox"], "lrrrrrr",
                 "Analisi di sensibilità sulla definizione del tempo di OS (scenario alla diagnosi).", "tab:sensibilita")]
    scrivi("extra_5_4.tex", "\n".join(t))


# ---------------------------------------------------------------- SHAP per variabile
def var_di(nome, vars_):
    if nome.startswith("missingindicator_"):
        nome = nome[len("missingindicator_"):]
    cand = [v for v in vars_ if nome == v or nome.startswith(v + "_")]
    return max(cand, key=len) if cand else nome


def valori_shap(model, Xt):
    """Valori SHAP (log-odds). Prova TreeExplainer; se la versione di SHAP non legge il modello XGBoost
    (errore tipo "could not convert string to float: '[3.5E-1]'") usa il calcolo nativo di XGBoost
    (pred_contribs), che da' gli stessi valori degli alberi."""
    try:
        import shap
        sv = shap.TreeExplainer(model).shap_values(Xt)
        if isinstance(sv, list):
            sv = sv[1]
        elif getattr(sv, "ndim", 2) == 3:
            sv = sv[:, :, 1]
        return np.asarray(sv)
    except Exception as e:
        if not hasattr(model, "get_booster"):
            raise
        print(f"   (SHAP non legge il modello: {type(e).__name__}; uso il calcolo nativo di XGBoost)")
        import xgboost as xgb
        c = model.get_booster().predict(xgb.DMatrix(np.asarray(Xt, dtype=float)), pred_contribs=True)
        return c[:, :-1]


def sez_shap():
    import shap
    sub, y = D["sub"], D["y"]
    imp, top = {}, {}
    for sk in ("Diag", "Post"):
        num, cat = get_columns(SC[sk], sub.columns)
        X = prepare_X(sub, num, cat)
        pipe = build_pipeline("XGBoost", num, cat, SEED).fit(X, y)
        pre = pipe.named_steps["pre"]; Xt = pre.transform(X); names = list(pre.get_feature_names_out())
        sv = valori_shap(pipe.named_steps["model"], Xt)
        gr = {}
        for j, n in enumerate(names):
            gr.setdefault(var_di(n, num + cat), []).append(j)
        agg = pd.Series({v: float(np.abs(sv[:, ix].sum(axis=1)).mean()) for v, ix in gr.items()}).sort_values(ascending=False)
        imp[sk] = agg; top[sk] = list(agg.index[:5])
        if sk == "Post":
            plt.figure(figsize=(9, 6)); shap.summary_plot(sv, pd.DataFrame(Xt, columns=names), max_display=15, show=False)
            plt.tight_layout(); plt.savefig(os.path.join(OUT, "shap_summary_post.pdf")); plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for a_, sk in zip(ax, ("Diag", "Post")):
        s = imp[sk].head(12)[::-1]; a_.barh([lab(x).replace("\\", "")[:38] for x in s.index], s.values, color="#2c3e50")
        a_.set_title(f"Scenario: {NOMI_SC[sk]}"); a_.set_xlabel("Media del valore assoluto SHAP (somma per variabile)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "shap_per_variabile.pdf")); plt.close()
    nuove = [v for v in top["Post"] if v not in top["Diag"]]
    t = ["\\subsection{Importanza per variabile}",
         "Poiché la codifica one-hot divide una variabile in più colonne, l'importanza SHAP è stata anche aggregata per variabile originale, sommando i valori "
         "SHAP delle colonne che le appartengono prima di calcolare il valore assoluto medio. La Figura~\\ref{fig:shapvar} mostra le prime dodici variabili nei due scenari.",
         "Alla diagnosi le cinque variabili con maggiore importanza aggregata sono: " + ", ".join(lab(v) for v in top["Diag"]) + ". Nello scenario post-induzione sono: "
         + ", ".join(lab(v) for v in top["Post"]) + (". Compaiono tra le prime cinque solo nel secondo scenario: " + ", ".join(lab(v) for v in nuove) + ".\n" if nuove else ".\n")
         + ("Il gruppo di rischio è assegnato in genere in base alle caratteristiche genetiche e alla risposta all'induzione, e determina l'intensità della terapia ricevuta: "
            "riassume quindi informazioni già presenti tra le altre variabili, e la sua elevata importanza va interpretata con cautela (da verificare sulle regole di assegnazione del protocollo).\n"
            if "Risk group" in top["Post"] else ""),
         figura("shap_per_variabile.pdf", "Importanza SHAP aggregata per variabile (XGBoost addestrato su tutti i dati dello scenario).", "fig:shapvar"),
         "\\subsection{Valori SHAP nello scenario post-induzione}",
         "La Figura~\\ref{fig:shappost} riporta i valori SHAP del modello XGBoost nello scenario post-induzione. Valgono le stesse cautele del paragrafo "
         "precedente: il modello è addestrato su tutti i dati e i valori descrivono associazioni usate dal modello, non relazioni di causa.\n",
         figura("shap_summary_post.pdf", "Valori SHAP del modello XGBoost nello scenario post-induzione (prime 15 colonne).", "fig:shappost", "0.85")]
    scrivi("extra_5_2.tex", "\n".join(t))


# ---------------------------------------------------------------- pazienti deceduti durante l'induzione
def sez_landmark():
    sub, y = D["sub"], D["y"]
    c = "CR status at end of course 1"
    if c not in sub.columns:
        print("   colonna assente"); return
    morti = sub[c].astype(str).str.lower().eq("death").to_numpy()
    n_m, n_md = int(morti.sum()), int(y[morti].sum())
    osm = float(sub.loc[morti, "Overall Survival Time in Days"].median())
    righe, res = [], {}
    for nome, mask in (("tutti i pazienti", np.ones(len(sub), bool)), ("esclusi i deceduti in induzione", ~morti)):
        s2, y2 = sub[mask], y[mask]
        for sk in ("Diag", "Post"):
            num, cat = get_columns(SC[sk], s2.columns); X = prepare_X(s2, num, cat)
            for mn in MODEL_NAMES:
                print(f"   {nome[:12]} / {sk} / {mn}")
                auc = roc_auc_score(y2, repeated_cv(build_pipeline(mn, num, cat, SEED), X, y2, n_repeats=A.ripetizioni, seed=SEED)["oof"])
                res[(nome, sk, mn)] = auc
                righe.append([nome, NOMI_SC[sk], ABBR[mn], f"{len(s2)}", f"{int(y2.sum())}", f"{auc:.3f}"])
    a0, a1 = res[("tutti i pazienti", "Post", "XGBoost")], res[("esclusi i deceduti in induzione", "Post", "XGBoost")]
    d0, d1 = res[("tutti i pazienti", "Diag", "XGBoost")], res[("esclusi i deceduti in induzione", "Diag", "XGBoost")]
    t = ["\\subsection{Pazienti deceduti durante l'induzione}",
         f"Nello scenario post-induzione la categoria ``death'' della risposta a fine primo corso identifica {m('NDeathInd', n_m, '{}')} pazienti deceduti durante l'induzione "
         + (f"(tutti deceduti entro 3 anni, con OS mediana di {m('OSDeathInd', osm, '{:.0f}')} giorni)" if n_md == n_m else
            f"({m('NDeathIndMorti', n_md, '{}')} deceduti entro 3 anni, con OS mediana di {m('OSDeathInd', osm, '{:.0f}')} giorni)")
         + ". Per questi pazienti la risposta alla terapia coincide con l'esito: il modello la apprende facilmente, come si vede dai valori SHAP elevati della "
         "Figura~\\ref{fig:shappost}. Per valutare quanto questo incida, le analisi sono state ripetute escludendo tali pazienti (Tabella~\\ref{tab:landmark}). "
         f"La ROC-AUC di XGBoost alla diagnosi {passa(d0, d1)} e nello scenario post-induzione {passa(a0, a1)}. "
         + ("La riduzione nello scenario post-induzione indica che una parte del vantaggio rispetto allo scenario alla diagnosi deriva da questi pazienti, "
            "e va considerata nella lettura dei risultati.\n" if (a0 - a1) > 0.01 else "L'esclusione non modifica in modo rilevante i risultati.\n"),
         tabella(righe, ["Pazienti inclusi", "Scenario", "Modello", "Pazienti", "Decessi", "ROC-AUC"], "lllrrr",
                 "ROC-AUC con e senza i pazienti deceduti durante l'induzione (predizioni out-of-fold).", "tab:landmark")]
    scrivi("extra_5_4b.tex", "\n".join(t))


SEZIONI = {"flusso": sez_flusso, "variabili": sez_variabili, "curve": sez_curve, "pesi": sez_pesi, "tuning": sez_tuning,
           "sopravvivenza": sez_sopravvivenza, "shap": sez_shap, "landmark": sez_landmark}


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--dati", default="data/clinical_supplements")
    ap.add_argument("--solo", nargs="*", default=list(SEZIONI))
    ap.add_argument("--ripetizioni", type=int, default=5, help="ripetizioni della CV 5-fold")
    ap.add_argument("--iter", type=int, default=8, help="combinazioni della ricerca casuale (tuning)")
    A = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    carica()
    for nome in A.solo:
        if nome not in SEZIONI:
            print("sezione sconosciuta:", nome); continue
        t0 = time.time(); print(f"[{time.strftime('%H:%M:%S')}] {nome}")
        try:
            SEZIONI[nome]()
        except Exception as e:  # una sezione che fallisce non ferma le altre
            print(f"   ERRORE in {nome}: {type(e).__name__}: {e}")
        salva_macro(); print(f"   fatto in {time.time() - t0:.0f} s")
    print("\nFinito. File in", OUT)


if __name__ == "__main__":
    main()