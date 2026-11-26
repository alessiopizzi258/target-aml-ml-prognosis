"""
aggiorna_risultati.py - Copia nella tesi i risultati esportati dall'app, senza modificarli a mano.

    python aggiorna_risultati.py C:\\percorso\\tesi_aml\\outputs

Copia risultati_macro.tex, tabella_risultati.tex (rendendola centrata e ridimensionata),
e le figure kaplan_meier_os.pdf / shap_summary.pdf se presenti.
"""
import os
import shutil
import sys

src = sys.argv[1] if len(sys.argv) > 1 else "../tesi_aml/outputs"
os.makedirs("risultati", exist_ok=True)
os.makedirs("figure", exist_ok=True)

shutil.copy(os.path.join(src, "risultati_macro.tex"), "risultati/risultati_macro.tex")
t = open(os.path.join(src, "tabella_risultati.tex"), encoding="utf-8").read().replace("\r", "")
t = t.replace("\\begin{table}\n", "\\begin{table}[htbp]\n\\centering\\small\n", 1)
t = t.replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}", 1)
t = t.replace("\\end{tabular}", "\\end{tabular}}", 1)
open("risultati/tabella_risultati.tex", "w", encoding="utf-8").write(t)
for nome, dest in (("kaplan_meier_os.pdf", "kaplan_meier_os.pdf"), ("shap_summary.pdf", "shap_summary_diagnosi.pdf")):
    p = os.path.join(src, nome)
    if os.path.exists(p):
        shutil.copy(p, os.path.join("figure", dest))
        print("copiato", nome)
print("Fatto. Ricompila con: pdflatex main && bibtex main && pdflatex main && pdflatex main")
print("ATTENZIONE: shap_summary.pdf va copiato solo se calcolato sullo scenario DIAGNOSI.")
