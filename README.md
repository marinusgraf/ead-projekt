# EAD-Projekt: TCGA PanCanAtlas

Abschlussprojekt im Kurs *Effiziente Algorithmen in Data Science und KI* (WS 2026/27).
Wir untersuchen Genexpressionsdaten von Tumoren mit Dimensionsreduktion (PCA und eine weitere Methode, z. B. UMAP)
und Clustering (spektrales Clustering und ein weiteres Verfahren).

## Daten

TCGA PanCanAtlas (*The Cancer Genome Atlas*, Pan-Cancer Atlas, 2018): RNA-Seq-Genexpression von **11.069 Proben aus 33 Tumortypen**
für **20.531 Gene**, batch-korrigiert mit EB++. Dazu kommen die klinischen Daten (TCGA-CDR) und Qualitätsangaben pro Probe.

Quelle: <https://gdc.cancer.gov/about-data/publications/pancanatlas> (frei zugänglich, kein Login)

| Datei | Inhalt |
|---|---|
| `EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv` | Genexpression, Gene × Proben (1,9 GB) |
| `TCGA-CDR-SupplementalTableS1.xlsx` | klinische Daten pro Patient: Tumortyp, Geschlecht, Alter, Überleben |
| `merged_sample_quality_annotations.tsv` | Qualitätsangaben, u. a. auszuschließende Proben |

Die Daten sind wegen ihrer Größe **nicht im Repository**. Das Notebook lädt sie beim ersten Ausführen automatisch nach `data/pancanatlas/`.

## Notebooks

| Notebook | Inhalt |
|---|---|
| `notebook_tcga_daten.ipynb` | Daten verstehen: Herkunft, Gene, Proben, Tumortypen, Werteverteilung, Probleme und offene Entscheidungen für die Aufbereitung |
| `notebook_pca_primaertumoren.ipynb` | Erster einfacher PCA-Durchlauf auf den Primärtumoren: Bilden die Tumortypen eigene Gruppen, und wo weichen die Gruppen von den Tumortypen ab? Mit offenen Fragen am Ende. Braucht die Ausgaben von `notebook_tcga_daten.ipynb`. |

## Einrichtung

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Beim ersten Lauf von `notebook_tcga_daten.ipynb` werden ca. 1,9 GB heruntergeladen und einmalig in eine schnellere Binärdatei umgewandelt
(ca. 0,9 GB, ebenfalls in `data/pancanatlas/`). Insgesamt werden etwa 3 GB Speicherplatz benötigt; empfohlen sind mindestens 8 GB Arbeitsspeicher.
