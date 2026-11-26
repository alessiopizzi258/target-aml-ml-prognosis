# Modelli predittivi di Machine Learning per il monitoraggio clinico in oncoematologia pediatrica

Repository ufficiale per la riproducibilita' degli esperimenti, delle tabelle e delle figure della tesi di laurea triennale in Ingegneria Informatica (Universita' Telematica eCampus, A.A. 2025/2026).

- **Autore**: Alessio Pizzi
- **Relatore**: Prof. Michele Bernardini

## 1. Dati originali (TARGET-AML)
Gli esperimenti utilizzano i file clinici open-access del programma **TARGET-AML** (studio dbGaP `phs000465`, *Clinical Supplement*, versione 20 luglio 2023), scaricabili dal portale [NCI Genomic Data Commons (GDC)](https://portal.gdc.cancer.gov/):
- `AML1031`
- `Discovery`
- `Validation`
- `LowDepthRNAseq`
- `AdditionalCases`

Inserire i file `.xlsx` scaricati in `data/clinical_supplements/`.

## 2. Installazione ambiente
Ambiente di riferimento: **Python 3.12.9** (seme casuale fisso `42`).

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Esecuzione delle analisi
```bash
python analisi_extra.py
python aggiorna_risultati.py outputs
```
