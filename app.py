"""App locale per la tesi TARGET-AML.  Avvio:  streamlit run app.py

L'app CALCOLA e ESPORTA risultati reali (.tex/.pdf). Non genera testo di tesi.
"""
import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data import load_raw, merge_patients, clean, binary_target, survival_target
from src.features import SCENARIOS, get_columns
from src.models import MODEL_NAMES, build_pipeline, prepare_X
from src.evaluate import repeated_cv, summarize
from src.survival import cv_cox, km_curve
from src.explain import shap_summary
from src.export_latex import write_macros, write_table

st.set_page_config(page_title="Tesi TARGET-AML", layout="wide")
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

st.title("TARGET-AML: predizione di mortalità, pipeline riproducibile")
folder = st.sidebar.text_input("Cartella dei Clinical Supplement", "data/clinical_supplements")
only20 = st.sidebar.checkbox("Solo coorte TARGET-20 (AML pediatrica standard)", value=False)
horizon = st.sidebar.number_input("Orizzonte target binario (giorni)", 365, 3650, 1095, step=365)
seed = st.sidebar.number_input("Seed", 0, 9999, 42)


@st.cache_data(show_spinner="Carico e unisco i file...")
def load(folder):
    raw = load_raw(folder)
    pat, conflicts = merge_patients(raw)
    return raw, pat, conflicts


try:
    raw, pat, conflicts = load(folder)
except Exception as e:
    st.error(f"Impossibile caricare i dati: {e}")
    st.stop()

d = clean(pat, only_target20=only20)
sub, y = binary_target(d, int(horizon))
if "results" not in st.session_state:
    st.session_state["results"] = {}

t_data, t_eda, t_mod, t_surv, t_shap, t_exp = st.tabs(
    ["1 Dati", "2 EDA", "3 Modelli", "4 Survival", "5 Spiegabilità", "6 Export"])

with t_data:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Righe nei file", len(raw))
    c2.metric("Pazienti unici", len(pat))
    c3.metric("Dopo filtri coorte", len(d))
    c4.metric(f"Usabili per target a {int(horizon)} gg", len(sub), f"{y.mean():.1%} decessi")
    st.subheader("Conflitti tra file (stesso paziente, valori diversi)")
    st.caption("Va dichiarato in tesi come è stato risolto (qui: primo valore non nullo).")
    st.dataframe(conflicts.head(15))
    st.subheader("Valori mancanti per colonna")
    miss = d.isna().mean().sort_values(ascending=False).head(25)
    fig, ax = plt.subplots(figsize=(8, 6)); miss[::-1].plot.barh(ax=ax, color="#2c3e50")
    ax.set_xlabel("Frazione di valori mancanti"); st.pyplot(fig)

with t_eda:
    ds, T, E = survival_target(d)
    st.subheader("Tabella 1: caratteristiche della coorte")
    t1 = pd.DataFrame({
        "Valore": [str(len(ds)), f"{ds['Age_years'].median():.1f} ({ds['Age_years'].quantile(.25):.1f}-{ds['Age_years'].quantile(.75):.1f})",
                   f"{(ds['Gender'] == 'Male').mean():.1%}", f"{ds['WBC at Diagnosis'].median():.1f}",
                   f"{E.mean():.1%}", f"{pd.Series(T).median():.0f}"]},
        index=["N pazienti", "Età mediana in anni (IQR)", "Maschi", "WBC mediano alla diagnosi",
               "Decessi", "Follow-up/OS mediano (giorni)"])
    st.table(t1)
    st.subheader("Kaplan-Meier (Overall Survival)")
    k = km_curve(T, E, "Coorte")
    fig, ax = plt.subplots(figsize=(8, 5)); k.plot_survival_function(ax=ax, color="#c0392b")
    ax.set_ylim(0, 1); ax.set_xlabel("Giorni dalla diagnosi"); ax.set_ylabel("Probabilità di sopravvivenza")
    ax.axvline(horizon, ls="--", color="gray"); st.pyplot(fig)
    if st.button("Salva KM in outputs/"):
        fig.savefig(f"{OUT}/kaplan_meier_os.pdf"); st.success("Salvato outputs/kaplan_meier_os.pdf")

with t_mod:
    st.caption("Imputazione ed encoding sono DENTRO ogni fold. Le metriche IC95% sono bootstrap sulle predizioni out-of-fold.")
    scens = st.multiselect("Scenari di feature", list(SCENARIOS), default=list(SCENARIOS))
    models = st.multiselect("Modelli", MODEL_NAMES, default=MODEL_NAMES)
    reps = st.slider("Ripetizioni CV 5-fold", 1, 10, 5)
    if st.button("Esegui", type="primary"):
        bar = st.progress(0.0); jobs = [(s, m) for s in scens for m in models]
        for j, (s, m) in enumerate(jobs):
            num, cat = get_columns(s, sub.columns)
            X = prepare_X(sub, num, cat)
            r = repeated_cv(build_pipeline(m, num, cat, int(seed)), X, y, n_repeats=reps, seed=int(seed),
                            progress=lambda p, j=j: bar.progress((j + p) / len(jobs)))
            st.session_state["results"][(s, m)] = summarize(r, y)
        bar.progress(1.0)
    if st.session_state["results"]:
        res = pd.DataFrame(st.session_state["results"]).T
        res.index.names = ["Scenario", "Modello"]
        st.dataframe(res.round(3))

with t_surv:
    st.caption("Cox penalizzato (L2). Usa tutti i pazienti, anche i censurati: nessuna esclusione come nel target binario.")
    ds, T, E = survival_target(d)
    scen = st.selectbox("Scenario", list(SCENARIOS)[1:], key="surv_s")
    pen = st.slider("Penalizzazione", 0.01, 1.0, 0.1)
    if st.button("Esegui Cox"):
        num, cat = get_columns(scen, ds.columns)
        c = cv_cox(prepare_X(ds, num, cat), T, E, num, cat, n_repeats=3, penalizer=pen, seed=int(seed))
        st.session_state["cox"] = {**st.session_state.get("cox", {}), scen: (c.mean(), c.std(), len(ds))}
    for s, (m, sd, n) in st.session_state.get("cox", {}).items():
        st.write(f"**{s}**: c-index {m:.3f} ± {sd:.3f} (N={n})")

with t_shap:
    ok_scen = sorted({s for (s, m), v in st.session_state["results"].items() if v["ROC-AUC (OOF)"] >= 0.6})
    if not ok_scen:
        st.warning("SHAP è disabilitato finché uno scenario non supera AUC 0.60: spiegare un modello che non predice significa spiegare rumore. Esegui prima il tab Modelli.")
    else:
        st.caption("Sono offerti solo gli scenari con AUC >= 0.60 nei risultati del tab Modelli.")
        scen = st.selectbox("Scenario", ok_scen, key="shap_s")
        mname = st.selectbox("Modello", [m for m in MODEL_NAMES if m != "Regressione Logistica"], key="shap_m")
        if st.button("Calcola SHAP"):
            num, cat = get_columns(scen, sub.columns)
            fig, imp = shap_summary(prepare_X(sub, num, cat), y, num, cat, mname, f"{OUT}/shap_summary.pdf")
            st.pyplot(fig); st.dataframe(imp.head(15).rename("mean |SHAP|"))
            st.caption("Salvato in outputs/shap_summary.pdf. Nota: SHAP descrive il modello, non causalità biologica.")

with t_exp:
    if st.session_state["results"]:
        res = pd.DataFrame(st.session_state["results"]).T
        res.index.names = ["Scenario", "Modello"]
        if st.button("Esporta .tex"):
            st.success(f"Scritti: {write_macros(res, f'{OUT}/risultati_macro.tex')}, {write_table(res, f'{OUT}/tabella_risultati.tex')}")
            st.code("\\input{outputs/risultati_macro.tex}\n% poi nel testo:  AUC pari a \\AUCDiagnosiXGBoost", language="latex")
    else:
        st.info("Esegui prima i modelli.")
