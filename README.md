# EAD-Projekt: TCGA PanCanAtlas

Abschlussprojekt im Kurs *Effiziente Algorithmen in Data Science und KI* (KI: Künstliche Intelligenz; Wintersemester 2026/27).
Wir untersuchen Genexpressionsdaten von Tumoren mit Dimensionsreduktion (Hauptkomponentenanalyse, PCA, *Principal Component Analysis*, und eine weitere Methode,
zum Beispiel UMAP, *Uniform Manifold Approximation and Projection*)
und Clustering (spektrales Clustering und ein weiteres Verfahren).

**Forschungsfrage:** Wie ähneln sich verschiedene Tumortypen in ihrer Genexpression?

## Daten

TCGA PanCanAtlas (*The Cancer Genome Atlas*, Pan-Cancer Atlas, 2018): Genexpression aus RNA-Sequenzierung (RNA: Ribonukleinsäure) von **11.069 Proben aus 33 Tumortypen**
für **20.531 Gene**, batch-korrigiert mit EB++ (*Empirical Bayes*, erweitert). Dazu kommen die klinischen Daten (Clinical Data Resource) und Qualitätsangaben pro Probe.

Quelle: <https://gdc.cancer.gov/about-data/publications/pancanatlas> (frei zugänglich, kein Login)

| Datei | Inhalt |
|---|---|
| `EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv` | Genexpression, Gene × Proben (1,9 Gigabyte) |
| `TCGA-CDR-SupplementalTableS1.xlsx` | klinische Daten pro Patient: Tumortyp, Geschlecht, Alter, Überleben |
| `merged_sample_quality_annotations.tsv` | Qualitätsangaben, unter anderem auszuschließende Proben |

Die Daten sind wegen ihrer Größe **nicht im Repository**. `notebook_aufbereitung.ipynb` lädt sie beim ersten Ausführen automatisch nach `data/pancanatlas/`.

## Dokumente und Notebooks

| Datei | Inhalt |
|---|---|
| `datenaufbereitung.md` | Erklärung des Datensatzes (Herkunft, Aufbau, Gene, Proben, gemessene Werte, Batch-Effekte) und alle Entscheidungen für die Aufbereitung mit Optionen und Vorschlägen |
| `notebook_aufbereitung.ipynb` | Lädt die Daten herunter, stellt die Metadaten zu Genen und Proben zusammen (Tumortyp, Geschlecht, Sequenziergerät, Qualitätsangaben), setzt die Entscheidungen aus `datenaufbereitung.md` als eine Funktion mit Parametern um, prüft die offenen Punkte (Geschlechtsgene, fehlende Werte, Platten- und Geräteeffekte, auffällige Proben, Gruppengrößen) und speichert die fertige Analysematrix (9.104 Primärtumoren × 5.000 Gene) |
| `notebook_pca.ipynb` | Hauptkomponentenanalyse (PCA, *Principal Component Analysis*) der aufbereiteten Primärtumoren: erklärte Varianz, Lage der Tumortypen in den ersten Hauptkomponenten, Bedeutung der Achsen (Gewebe- und Zellprogramme). Speichert die Hauptkomponenten. |
| `umap_erklaerung.md` | Erklärung von UMAP: Nachbarschaftsgraph, lokale Gewichte, Symmetrisieren, spektrale Einbettung als Start, Optimierung, Parameter und was man aus dem Bild ablesen darf |
| `notebook_umap.ipynb` | Baut auf den ersten 50 Hauptkomponenten auf: Welche Tumortypen ähneln sich (Nachbarschaften)? UMAP-Bild, Tumortypen, die in histologische Untertypen zerfallen, und Robustheit gegenüber Parametern, Eingabe und Zufallsstartwert |

**Reihenfolge:** zuerst `notebook_aufbereitung.ipynb` (lädt die Daten und bereitet sie auf), dann `notebook_pca.ipynb`, dann `notebook_umap.ipynb`.
Jedes Notebook verwendet Dateien, die die vorherigen in `data/pancanatlas/` gespeichert haben:

| Datei | erzeugt von | Inhalt |
|---|---|---|
| `aufbereitet_matrix.npy` | `notebook_aufbereitung.ipynb` | Proben × Gene, log2(x + 1), zentriert |
| `aufbereitet_proben.csv`, `aufbereitet_gene.csv` | `notebook_aufbereitung.ipynb` | Metadaten in Zeilen- beziehungsweise Spaltenreihenfolge |
| `aufbereitet_parameter.json` | `notebook_aufbereitung.ipynb` | Parameter der Aufbereitung, Protokoll der Schritte |
| `pca_scores.npy`, `pca_ladungen.npy`, `pca_erklaerte_varianz.npy` | `notebook_pca.ipynb` | Koordinaten der Proben, Ladungen und erklärte Varianz der ersten 100 Hauptkomponenten |
| `umap_2d.npy`, `nachbarn_pca50.npy` | `notebook_umap.ipynb` | UMAP-Einbettung; die 15 nächsten Nachbarn jeder Probe in den ersten 50 Hauptkomponenten |

## Einrichtung

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Beim ersten Lauf von `notebook_aufbereitung.ipynb` werden etwa 1,9 Gigabyte heruntergeladen und einmalig in eine schnellere Binärdatei umgewandelt
(etwa 0,9 Gigabyte, ebenfalls in `data/pancanatlas/`). Die aufbereiteten Dateien brauchen weitere etwa 0,2 Gigabyte. Insgesamt werden etwa 3 Gigabyte Speicherplatz benötigt;
empfohlen sind mindestens 8 Gigabyte Arbeitsspeicher, für `notebook_aufbereitung.ipynb` eher 16 Gigabyte (es hält mehrere Varianten der Matrix gleichzeitig im Speicher).
