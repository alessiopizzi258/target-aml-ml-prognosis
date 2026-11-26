"""Caricamento e unione dei Clinical Supplement TARGET-AML (GDC)."""
import glob
import os
import numpy as np
import pandas as pd

ID = "TARGET USI"
# Foglio con i dati per ogni tipo di file (gli altri xlsx della cartella vengono ignorati)
PATTERNS = {
    "AML1031": "*ClinicalData_AML1031_*.xlsx",
    "Discovery": "*ClinicalData_Discovery_2*.xlsx",
    "Validation": "*ClinicalData_Validation_*.xlsx",
    "LowDepthRNAseq": "*ClinicalData_LowDepthRNAseq_*.xlsx",
    "AdditionalCases": "*additionalCasesForSortedCells*.xlsx",
}


def load_raw(folder: str) -> pd.DataFrame:
    """Concatena tutti i file clinici, aggiungendo la colonna 'source_file'."""
    parts = []
    for label, pat in PATTERNS.items():
        for path in glob.glob(os.path.join(folder, pat)):
            xl = pd.ExcelFile(path)
            sheet = "Clinical Data" if "Clinical Data" in xl.sheet_names else xl.sheet_names[0]
            d = xl.parse(sheet)
            d["source_file"] = label
            parts.append(d)
    if not parts:
        raise FileNotFoundError(f"Nessun file clinico trovato in {folder}")
    raw = pd.concat(parts, ignore_index=True)
    raw = raw.dropna(subset=[ID])
    return raw


def merge_patients(raw: pd.DataFrame):
    """Un paziente = una riga.

    Per ogni colonna prende il primo valore non nullo tra i file in cui il
    paziente compare. Restituisce anche il report dei conflitti (stesso
    paziente, valori diversi in file diversi).
    """
    data_cols = [c for c in raw.columns if c not in (ID, "source_file")]
    g = raw.groupby(ID)
    merged = g[data_cols].first()          # primo non-null per colonna
    n_files = g.size().rename("n_files")
    n_distinct = g[data_cols].nunique(dropna=True)
    conflicts = (n_distinct > 1).sum().sort_values(ascending=False)
    conflicts = conflicts[conflicts > 0].rename("pazienti_con_valori_discordanti")
    merged = merged.join(n_files)
    return merged.reset_index(), conflicts


def clean(patients: pd.DataFrame, only_target20: bool = False) -> pd.DataFrame:
    d = patients.copy()
    if only_target20:
        d = d[d[ID].str.startswith("TARGET-20")]
    # Normalizza maiuscole/minuscole ("Induction Failure" vs "Induction failure")
    d["First Event"] = d["First Event"].astype("string").str.strip().str.capitalize()
    d["Vital Status"] = d["Vital Status"].astype("string").str.strip().str.capitalize()
    num_cols = [
        "Age at Diagnosis in Days", "WBC at Diagnosis",
        "Bone marrow leukemic blast percentage (%)", "Peripheral blasts (%)",
        "Overall Survival Time in Days", "Event Free Survival Time in Days",
        "MRD % at end of course 1", "MRD % at end of course 2",
    ]
    for c in num_cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["Age_years"] = d["Age at Diagnosis in Days"] / 365.25
    return d


def valid_os(d: pd.DataFrame) -> pd.DataFrame:
    """Pazienti con stato vitale e tempo di OS noti."""
    return d.dropna(subset=["Vital Status", "Overall Survival Time in Days"]).copy()


def binary_target(d: pd.DataFrame, horizon_days: int = 1095):
    """Target: decesso entro `horizon_days`.

    Esclusi i pazienti censurati prima dell'orizzonte (esito non noto).
    Restituisce (sottoinsieme, y).
    """
    d = valid_os(d)
    t = d["Overall Survival Time in Days"]
    dead = d["Vital Status"].eq("Dead")
    keep = (t > horizon_days) | (dead & (t <= horizon_days))
    sub = d[keep].copy()
    y = (sub["Vital Status"].eq("Dead") & (sub["Overall Survival Time in Days"] <= horizon_days)).astype(int).values
    return sub, y


def survival_target(d: pd.DataFrame):
    """Tempo ed evento per la survival analysis (nessun paziente escluso)."""
    d = valid_os(d)
    d = d[d["Overall Survival Time in Days"] > 0].copy()
    return d, d["Overall Survival Time in Days"].values, d["Vital Status"].eq("Dead").astype(int).values
