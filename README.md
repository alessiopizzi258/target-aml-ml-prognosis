# Modelli predittivi di Machine Learning per il monitoraggio clinico in oncoematologia pediatrica

Repository ufficiale per la riproducibilita' degli esperimenti, delle tabelle e delle figure della tesi di laurea triennale in Ingegneria Informatica (Universita' Telematica eCampus, A.A. 2025/2026).

- **Autore**: Alessio Pizzi
- **Relatore**: Prof. Michele Bernardini

## 1. Dati originali (TARGET-AML)
Gli esperimenti utilizzano i 5 file clinici open-access del programma **TARGET-AML** (studio dbGaP `phs000465`, *Clinical Supplement*, versione 20 luglio 2023), scaricabili dal portale [NCI Genomic Data Commons (GDC)](https://portal.gdc.cancer.gov/):
- `AML1031`
- `Discovery`
- `Validation`
- `LowDepthRNAseq`
- `AdditionalCases`

Posizionare i file `.xlsx` originali all'interno della cartella `data/`.

## 2. Installazione ambiente
Testato con **Python 3.12.9**.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Riproduzione dei risultati
Con seme casuale fisso `42`:

```bash
python aggiorna_risultati.py
python analisi_extra.py
```
