"""Definizione degli scenari di feature.

SCENARIO 'diagnosi'   : solo informazioni disponibili al momento della diagnosi.
SCENARIO 'post_induzione': aggiunge risposta al trattamento (MRD, CR, risk group).
                           NON è una predizione alla diagnosi: va dichiarato in tesi.

Escluse SEMPRE (sono l'esito o ne derivano): First Event, Event Free Survival Time,
Year of Last Follow Up, siti di recidiva, MRD/CR dopo il corso 2, SCT in 1st CR.
"""

DIAG_NUM = [
    "Age_years",
    "WBC at Diagnosis",
    "Bone marrow leukemic blast percentage (%)",
    "Peripheral blasts (%)",
]

DIAG_CAT = [
    "Gender", "Race", "Ethnicity", "CNS disease", "Chloroma", "FAB Category",
    "t(6;9)", "t(8;21)", "t(3;5)(q25;q34)", "t(6;11)(q27;q23)", "t(9;11)(p22;q23)",
    "t(10;11)(p11.2;q23)", "t(11:19)(q23:p13.1)", "inv(16)",
    "del5q", "del7q", "del9q", "monosomy 5", "monosomy 7", "trisomy 8", "trisomy 21",
    "MLL", "Minus Y", "Minus X",
    "Primary Cytogenetic Code", "Cytogenetic Complexity",
    "FLT3/ITD positive?", "FLT3 PM", "NPM mutation", "CEBPA mutation", "WT1 mutation",
    "c-Kit Mutation Exon 8", "c-Kit Mutation Exon 17",
]

POST_NUM = ["MRD % at end of course 1"]
POST_CAT = ["MRD at end of course 1", "CR status at end of course 1", "Risk group"]

SCENARIOS = {
    "Solo età e sesso (baseline)": (["Age_years"], ["Gender"]),
    "Diagnosi (tutte le feature cliniche)": (DIAG_NUM, DIAG_CAT),
    "Post-induzione (+ MRD, CR1, risk group)": (DIAG_NUM + POST_NUM, DIAG_CAT + POST_CAT),
}


def get_columns(scenario: str, available):
    num, cat = SCENARIOS[scenario]
    num = [c for c in num if c in available]
    cat = [c for c in cat if c in available]
    return num, cat
