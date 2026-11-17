# Datenaufbereitung: Datensatz und offene Entscheidungen

Dieses Dokument fasst zusammen, **was im Datensatz steckt** (Teil 1) und **welche Entscheidungen wir für die Aufbereitung treffen müssen** (Teil 2),
bevor Hauptkomponentenanalyse (PCA, *Principal Component Analysis*), UMAP (*Uniform Manifold Approximation and Projection*) und Clustering belastbare Ergebnisse liefern.
**Forschungsfrage:** Wie ähneln sich verschiedene Tumortypen in ihrer Genexpression?
Die Zahlen stammen aus der ersten Erkundung der Daten (früher `notebook_tcga_daten.ipynb`, noch in der Git-Historie) und aus den Notebooks.
Laden und Aufbereiten der Daten erledigt jetzt `notebook_aufbereitung.ipynb`.

---

## Teil 1: Der Datensatz

### 1.1 Herkunft

- **TCGA** (*The Cancer Genome Atlas*, „Krebsgenom-Atlas“): US-amerikanisches Forschungsprogramm (2006–2018), das über 11.000 Tumoren aus 33 Krebsarten molekular vermessen hat.
- **PanCanAtlas** (*Pan-Cancer Atlas*, 2018): die abschließende gemeinsame Auswertung aller Krebsarten; alle Daten wurden dafür einheitlich aufbereitet.
- Quelle: GDC (*Genomic Data Commons*) des NCI (*National Cancer Institute*, Nationales Krebsforschungsinstitut der USA),
  <https://gdc.cancer.gov/about-data/publications/pancanatlas>, frei zugänglich.

| Datei | Inhalt |
|---|---|
| `EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv` | Genexpression, Gene $\times$ Proben, batch-korrigiert (1,9 Gigabyte) |
| `TCGA-CDR-SupplementalTableS1.xlsx` | Clinical Data Resource (CDR, kuratierte klinische Daten) pro **Patient**: Tumortyp, Geschlecht, Alter, Stadium, Histologie, Überleben |
| `merged_sample_quality_annotations.tsv` | Qualitätsangaben pro **Probe**, unter anderem Sequenziergerät und `Do_not_use` („nicht verwenden“) |

### 1.2 Aufbau im Überblick

Die Expressionsdatei ist eine große Tabelle (Matrix). Wir drehen sie beim Einlesen so, dass gilt:

- **eine Zeile pro Probe** (11.069 Proben, meist Tumorgewebe),
- **eine Spalte pro Gen** (20.531 Gene),
- **jeder Eintrag**: wie aktiv dieses Gen in dieser Probe ist (Genexpression, siehe 1.5).

Was zu einer Probe gehört (Tumortyp, Geschlecht, Alter), steht nicht in der Matrix, sondern in den beiden anderen Dateien; verknüpft wird über den Probennamen (TCGA-Barcode, siehe 1.4).

| | Anzahl | Bemerkung |
|---|---|---|
| Proben | 11.069 | 10.274 Patienten, 713 Kliniken, 241 Platten, 4 Sequenzierzentren |
| Gene | 20.531 | 29 ohne Symbol, ein doppeltes Symbol |
| Tumortypen | 33 | von 45 (Gallengangkrebs) bis 1.218 (Brustkrebs) Proben |
| fehlende Werte (NaN, *Not a Number*) | 1,7 % der Einträge | 1.751 Proben, 4.196 Gene betroffen |
| negative Werte | 1,25 Millionen | zwischen $-1$ und $0$ |

Die Matrix ist **breit** ($p \approx 2n$, fast doppelt so viele Gene wie Proben): die typische Situation für Dimensionsreduktion.

### 1.3 Die Gene

Jede Spalte ist ein Gen, benannt in der Form `Symbol|Entrez-Nummer`, zum Beispiel `TP53|7157` (Tumorprotein p53).

- Das **Symbol** ist der übliche Kurzname des Gens. Die **Entrez-Nummer** ist die eindeutige Nummer in der Gendatenbank des National Center for Biotechnology Information (Nationales Zentrum für Biotechnologie-Information der USA).
- **29 Gene haben kein Symbol** (`?`), nur eine Entrez-Nummer: Sie hatten zum Zeitpunkt der Annotation noch keinen offiziellen Namen.
- **Ein Symbol kommt doppelt vor** (`SLC35E2`, ein Membrantransporter), mit zwei verschiedenen Entrez-Nummern. Die Entrez-Nummern selbst sind eindeutig.
- Für lesbare Namen verwenden wir das Symbol und hängen nur in diesen Sonderfällen die Entrez-Nummer an.
- Die Annotation stammt etwa aus dem Jahr 2009. Manche Gene heißen deshalb noch anders als heute (zum Beispiel `SILV` statt `PMEL`).
- Die Datei enthält **keine weiteren Angaben** zu den Genen, insbesondere nicht, auf welchem Chromosom sie liegen.

**Einige Beispielgene**, die uns immer wieder begegnen:

| Gensymbol | ausgeschrieben | Rolle | erwartetes Muster |
|---|---|---|---|
| `GAPDH` | Glycerinaldehyd-3-phosphat-Dehydrogenase | Zuckerstoffwechsel | **Haushaltsgen**: in jeder Zelle hoch, kaum Unterschiede |
| `ACTB` | Beta-Aktin | Zellgerüst | Haushaltsgen |
| `ALB` | Albumin | Bluteiweiß, wird in der Leber gebildet | nur in Lebergewebe hoch |
| `KRT5` | Keratin 5 | Strukturprotein von Plattenepithel (Haut, Schleimhaut) | hoch in Plattenepithelkarzinomen |
| `ESR1` | Östrogenrezeptor 1 | Andockstelle für das Hormon Östrogen | hoch in hormonabhängigen Geweben (Brust, Gebärmutter) |
| `XIST` | *X-inactive specific transcript* | schaltet eines der beiden X-Chromosomen ab | nur in weiblichen Zellen aktiv |
| `RPS4Y1` | ribosomales Protein S4, Y-chromosomal | Baustein der Ribosomen, liegt auf dem Y-Chromosom | nur in männlichen Zellen aktiv |

### 1.4 Die Proben

Jede Probe trägt einen **TCGA-Barcode** (*The Cancer Genome Atlas*), zum Beispiel `TCGA-OR-A5J1-01A-11R-A29S-07`. Daraus lassen sich ablesen:
Klinik (`OR`), Patient (`TCGA-OR-A5J1`), **Probentyp** (`01`), **Platte** (`A29S`) und **Sequenzierzentrum** (`07`).

**Probentypen:**

| Code | Bedeutung | Proben |
|---|---|---|
| `01` | Primärtumor | 9.706 |
| `03` | Primärtumor aus Blut (nur akute myeloische Leukämie) | 173 |
| `11` | Normalgewebe (meist vom selben Patienten wie der Tumor) | 737 |
| `06` / `07` | Metastase (vor allem Melanom der Haut) | 396 |
| `02` | Rezidiv | 46 |
| `05` | zusätzlicher Primärtumor | 11 |

785 Patienten haben mehr als eine Probe, meist ein Paar aus Tumor und Normalgewebe.

**Tumortypen:** Der Tumortyp steht nicht in der Expressionsdatei; er kommt aus der klinischen Tabelle (Clinical Data Resource). 46 Proben ohne Eintrag dort bekommen ihn aus der Qualitätsdatei,
ohne Widersprüche. Die 33 Tumortypen mit ihren Kürzeln:

| Kürzel | Tumortyp | Kürzel | Tumortyp |
|---|---|---|---|
| ACC | Nebennierenrindenkarzinom | LUAD | Adenokarzinom der Lunge |
| BLCA | Harnblasenkarzinom | LUSC | Plattenepithelkarzinom der Lunge |
| BRCA | Brustkrebs | MESO | Mesotheliom |
| CESC | Gebärmutterhalskrebs | OV | seröses Ovarialkarzinom (Eierstock) |
| CHOL | Gallengangkrebs | PAAD | Bauchspeicheldrüsenkrebs |
| COAD | Dickdarmkrebs | PCPG | Phäochromozytom und Paragangliom |
| DLBC | diffus großzelliges B-Zell-Lymphom | PRAD | Prostatakrebs |
| ESCA | Speiseröhrenkrebs | READ | Mastdarmkrebs |
| GBM | Glioblastom (Hirntumor) | SARC | Sarkom (Weichteiltumor) |
| HNSC | Plattenepithelkarzinom von Kopf und Hals | SKCM | Melanom der Haut |
| KICH | chromophobes Nierenzellkarzinom | STAD | Magenkrebs |
| KIRC | klarzelliges Nierenzellkarzinom | TGCT | Keimzelltumor des Hodens |
| KIRP | papilläres Nierenzellkarzinom | THCA | Schilddrüsenkarzinom |
| LAML | akute myeloische Leukämie | THYM | Thymom |
| LGG | niedriggradiges Gliom (Hirntumor) | UCEC | Endometriumkarzinom (Gebärmutterkörper) |
| LIHC | Leberzellkarzinom | UCS | Karzinosarkom der Gebärmutter |
| | | UVM | Aderhautmelanom (Auge) |

Einige Typen sind biologisch eng verwandt: Niere (klarzellig, papillär, chromophob), Gehirn (niedriggradiges Gliom, Glioblastom), Darm (Dick- und Mastdarm),
Plattenepithelkarzinome (Lunge, Kopf-Hals, Teile von Gebärmutterhals und Speiseröhre), Gebärmutter (Endometriumkarzinom, Karzinosarkom), Melanome (Haut, Auge).

### 1.5 Die Werte: Was wird gemessen?

Gemessen wird die **Genexpression**: wie viel Boten-RNA (mRNA, *messenger RNA*; RNA = Ribonukleinsäure) jedes Gens in einer Gewebeprobe vorhanden ist.
Ein Gen, das abgelesen wird, um ein Protein herzustellen, erzeugt Boten-RNA; viel Boten-RNA heißt also: Das Gen ist **aktiv**.
Eine Probe ist damit ein Stück Gewebe (meist Tumor), beschrieben durch 20.531 Zahlen, eine pro Gen.

#### Vom Gewebe zur Zahl

1. **Gewebe und RNA:** Aus einem Stück Tumorgewebe wird die gesamte RNA gewonnen. Das Gewebe ist ein **Gemisch** aus Tumorzellen, Immunzellen,
   Bindegewebe und Blutgefäßen. Jeder Wert ist deshalb ein **Durchschnitt über alle Zellen** der Probe, nicht der Wert einer einzelnen Tumorzelle.
2. **Sequenzierung (RNA-Seq, RNA-Sequenzierung):** Die RNA wird in kurze Stücke zerlegt, und ein Sequenziergerät (Illumina HiSeq, bei frühen Proben der ältere
   Illumina Genome Analyzer) liest pro Probe einige zehn Millionen solcher Stücke aus. Die gelesenen Stücke heißen **Reads**.
3. **Zuordnung zu Genen mit RSEM** (*RNA-Seq by Expectation-Maximization*, RNA-Sequenzierung mit dem Erwartungsmaximierungs-Algorithmus):
   Jeder Read wird mit dem menschlichen Genom abgeglichen und einem Gen zugeordnet. Reads, die zu mehreren Genen passen, verteilt RSEM anteilig.
   Ergebnis ist eine **geschätzte Anzahl Reads pro Gen**. Durch die anteilige Verteilung sind die Werte nicht ganzzahlig (über 99 % der Werte größer als 0 haben Nachkommastellen).
4. **Upper-Quartile-Normalisierung** (Normalisierung auf das obere Quartil): Proben wurden unterschiedlich tief sequenziert (mal mehr, mal weniger Reads insgesamt).
   Damit Proben vergleichbar sind, wurde jede Probe durch das obere Quartil (75-%-Perzentil) ihrer Werte größer als 0 geteilt und mit 1.000 multipliziert.
   Nach diesem Schritt liegt das obere Quartil jeder Probe genau bei 1.000.
5. **Batch-Korrektur mit EB++** (*Empirical Bayes*, erweitert; Weiterentwicklung des Verfahrens ComBat): Proben vom älteren Gerät (Illumina Genome Analyzer)
   und aus anderen Sequenzierzentren hatten systematische Messunterschiede. Diese wurden **pro Gen** herausgerechnet, indem die Werte der betroffenen Proben
   verschoben und gestreckt wurden. Proben vom Illumina HiSeq aus dem Hauptzentrum (University of North Carolina) blieben unverändert.
   **Nicht** korrigiert wurden Unterschiede zwischen Platten (Gruppen von Proben, die gemeinsam verarbeitet wurden).

#### Was bedeutet eine einzelne Zahl?

- **Relativ, nicht absolut.** Eine Zahl ist keine Anzahl von Molekülen, sondern die Menge an Reads eines Gens **im Verhältnis zur Probe**.
  Der Wert 1.000 heißt: so viel wie das obere Quartil der exprimierten Gene dieser Probe, also ein gut exprimiertes Gen.
- **Typische Größenordnungen** (in Proben, die die Batch-Korrektur nicht verändert hat; nur Werte größer als 0):

  | Anteil der Werte | liegt unter |
  |---|---|
  | 10 % | 3 |
  | 25 % | 33 |
  | 50 % (Median) | 306 |
  | 75 % (oberes Quartil) | 1.000 |
  | 90 % | 2.388 |
  | 99 % | 13.167 |

  Der höchste Wert einer Probe liegt im Median bei etwa 290.000, im gesamten Datensatz bei fast 10 Millionen.
  Alle Zahlen hier sind Werte **nach** der Normalisierung auf das obere Quartil, so wie sie in der Datei stehen. Die Rohzählungen der Reads sind in unseren Dateien nicht enthalten.
  Die Normalisierung multipliziert jede Probe nur mit einem Faktor (1.000 geteilt durch ihr oberes Quartil); sie verschiebt das Niveau, verkleinert aber nicht die Spannweite.
  Die am stärksten exprimierten Gene sind **Haushaltsgene**, die jede Zelle braucht: `ACTB` (Beta-Aktin, Zellgerüst, Median etwa 97.000), `EEF1A1` (Elongationsfaktor 1-alpha 1, Proteinherstellung),
  `GAPDH` (Glycerinaldehyd-3-phosphat-Dehydrogenase, Zuckerstoffwechsel, Median etwa 60.000). Die 100 stärksten Gene einer Probe machen rund 22 % der Summe aller Werte aus.
- **Die Werte sind nicht nach Genlänge korrigiert.** Ein langes Gen liefert bei gleicher Aktivität mehr Reads als ein kurzes. Die Summe pro Probe liegt bei etwa 20 Millionen;
  es sind also keine Werte wie TPM (*Transcripts Per Million*, Transkripte pro Million), die längenkorrigiert sind und sich pro Probe zu 1 Million summieren.
  Folge: **Dasselbe Gen über verschiedene Proben** lässt sich direkt vergleichen, **verschiedene Gene in derselben Probe** nicht.
  Für unsere Analysen ist das unproblematisch: Nach der Log-Transformation ist der Längeneffekt pro Gen eine Konstante, die beim Zentrieren pro Gen verschwindet.
- **Null heißt „kein Read gefunden“.** 12,9 % aller Einträge sind exakt 0. Das Gen ist entweder inaktiv oder so schwach exprimiert, dass bei dieser Sequenziertiefe kein Read gefunden wurde.
  Rund 4 % der Einträge liegen zwischen 0 und 1.
- **Kleine Werte sind ungenau.** Die Werte entsprechen grob einer Zahl gezählter Reads. Beim Zählen seltener Ereignisse schwankt das Ergebnis etwa um die Wurzel der Anzahl:
  Bei 3 Reads sind das rund $\pm 1{,}7$, also über 50 %, bei 1.000 Reads nur rund 3 %. Gene mit durchgehend kleinen Werten bestehen deshalb überwiegend aus Rauschen.
- **Schiefe Verteilung.** Wenige Gene sind extrem hoch, die meisten niedrig: Die Verteilung ist stark rechtsschief (Schiefe 217).
  Erst nach $\log_2(x+1)$ wird sie annähernd symmetrisch; dann bedeutet ein Unterschied von 1 eine Verdopplung.
- **Nach der Batch-Korrektur** sind die Werte der korrigierten Proben keine Read-Zählungen mehr: Das obere Quartil liegt nicht mehr genau bei 1.000
  (Spanne etwa 780 bis 1.130), es gibt **negative Werte** (zwischen $-1$ und 0) und **fehlende Werte** (NaN, *Not a Number*) für Gene, bei denen EB++ keine Korrektur schätzen konnte.

#### Beispiel

Fünf Proben aus dem Nebennierenrindenkarzinom (ACC, *Adrenocortical Carcinoma*), gerundet:

| Probe | `GAPDH` | `ACTB` | `ALB` (Albumin, Leber) | `KRT5` (Keratin 5, Plattenepithel) | `XIST` (nur weiblich aktiv) | `RPS4Y1` (Y-Chromosom, nur männlich) |
|---|---|---|---|---|---|---|
| 1 | 57.986 | 68.807 | 1,4 | 0 | 0,5 | 7.014 |
| 2 | 45.123 | 71.065 | 5,2 | 0,8 | 7.993 | 5,2 |
| 3 | 270.999 | 73.505 | 5,3 | 1,2 | 194,6 | 5,3 |
| 4 | 127.711 | 111.056 | 312,9 | 0 | 0 | 6.803 |
| 5 | 45.653 | 52.458 | 1,1 | 0 | 2.109 | 1,1 |

Haushaltsgene liegen bei Zehn- bis Hunderttausenden, gewebefremde Gene (Leber, Plattenepithel) nahe 0, und von den Geschlechtsgenen ist je nach Geschlecht nur eines hoch.

#### Die Werte über alle Gene

- 219 Gene sind in keiner Probe exprimiert, 1.092 Gene in weniger als 10 % der Proben, 2.862 Gene haben eine mittlere log-Expression unter 1.
- Die Streuung der Gene fällt nach den ersten paar Tausend deutlich ab. Am stärksten streuen gewebespezifische Gene (`ALB` Albumin, Leber; `KRT5` Keratin 5, Plattenepithel)
  und Geschlechtsgene (`XIST`, *X-inactive specific transcript*; `RPS4Y1`, ribosomales Protein S4 vom Y-Chromosom). Haushaltsgene (`GAPDH`, `ACTB`) sind hoch, streuen aber kaum.

### 1.6 Bekannte Probleme

#### Batch-Effekte

**Was ist ein Batch-Effekt?** Ein *Batch* (deutsch etwa „Charge“) ist eine Gruppe von Proben, die gemeinsam verarbeitet wurde: in derselben Klinik entnommen,
auf derselben Platte aufbereitet, mit derselben Charge an Chemikalien, auf demselben Gerät, im selben Sequenzierzentrum, zur selben Zeit.
Jeder dieser Schritte hinterlässt kleine **systematische** Spuren in den Messwerten, zum Beispiel weil ein Gerät bestimmte RNA-Stücke etwas besser erfasst
oder weil Proben unterschiedlich lange gelagert wurden. Diese Spuren sind für alle Proben eines Batches gleich und haben **nichts mit der Biologie** zu tun.

**Warum ist das ein Problem?** Proben aus demselben Batch ähneln sich dadurch etwas mehr als Proben aus verschiedenen Batches.
Eine Hauptkomponentenanalyse oder ein Clustering sieht nur Ähnlichkeit und kann nicht unterscheiden, ob sie aus der Biologie oder aus dem Labor stammt.
Im schlimmsten Fall findet es „Gruppen“, die einfach Platten, Geräte oder Zentren sind.

**Bei TCGA ist der Batch mit dem Tumortyp vermengt.** Die Proben wurden meist tumortypweise gesammelt und verarbeitet:

| Batch-Ebene | Anzahl | Verteilung |
|---|---|---|
| Platte | 241 | 133 Platten (55 %) enthalten nur einen Tumortyp; ein Tumortyp verteilt sich im Median auf 16 Platten |
| Sequenzierzentrum (Code im Barcode) | 4 | `07` University of North Carolina: 9.772 Proben, alle Tumortypen · `13` British Columbia Genome Sciences Centre: Ovarialkarzinom, Magenkrebs, akute myeloische Leukämie (719) · `31`: Magenkrebs, Speiseröhrenkrebs, wenige Ovarialkarzinome (409) · `01` Broad Institute: nur Glioblastom (169) |
| Sequenziergerät | 2 | Illumina HiSeq: 10.438 Proben · älterer Illumina Genome Analyzer: 598 Proben, nur Endometriumkarzinom (366), Dickdarmkrebs (166) und Mastdarmkrebs (66) · ohne Angabe: 33 (Ovarialkarzinom) |

Unterscheiden sich zum Beispiel Speiseröhrenkrebs und Brustkrebs, kann das Biologie sein oder daran liegen, dass sie in verschiedenen Zentren sequenziert wurden.
Aus den Daten allein lässt sich das nicht vollständig trennen.

**Was die PanCanAtlas-Autoren schon korrigiert haben (EB++).** EB++ (*Empirical Bayes*, erweitert) ist eine Weiterentwicklung des Standardverfahrens ComBat:

- Eine Gruppe von Proben dient als **Referenz** und bleibt unverändert. Für jedes Gen wird geschätzt, um wie viel die Werte eines anderen Batches im Mittel **verschoben**
  und wie stark sie **gestreckt** sind, und die Werte dieses Batches werden entsprechend zurückgerechnet.
- „Empirical Bayes“ heißt: Die Schätzungen für ein einzelnes Gen werden zu den Schätzungen aller anderen Gene hin gezogen. Das macht sie auch bei wenigen Proben stabil.

Welche Proben tatsächlich verändert wurden, lässt sich an den Daten ablesen (verändert heißt: oberes Quartil nicht mehr bei 1.000, fehlende oder negative Werte):

| Gruppe | Proben | von EB++ verändert? |
|---|---|---|
| University of North Carolina, Illumina HiSeq | 9.174 | **nein, Referenz** (Ausnahme: 33 Prostatakrebs-Proben) |
| University of North Carolina, Illumina Genome Analyzer | 598 | **ja**, alle (Gerätekorrektur) |
| Zentren `13` und `31` | 1.095 | **ja**, fast alle (Zentrumskorrektur) |
| ohne Geräteangabe | 33 | ja |
| Broad Institute (Glioblastom) | 169 | **nein** |

**Was nicht korrigiert wurde:**
- Unterschiede zwischen **Platten** und zwischen **Kliniken**, die das Gewebe geliefert haben,
- die 169 Glioblastom-Proben vom Broad Institute,
- alles, was zeitlich bedingt ist (Lagerdauer, Wechsel von Chemikalien-Chargen).

**Grenzen der Korrektur.** Zentren `13` und `31` haben fast nur Tumortypen sequenziert, die im Referenzzentrum kaum vorkommen (akute myeloische Leukämie gar nicht).
Wenn ein Batch nur aus anderen Tumortypen besteht, ist eine Verschiebung pro Gen teilweise Batch-Effekt und teilweise echte Biologie. Wie EB++ das getrennt hat,
lässt sich aus unseren Dateien nicht ablesen; die Korrektur kann also auch **etwas echte Biologie entfernt** oder **Batch-Effekt übrig gelassen** haben.

**Nebenwirkungen der Korrektur:**
- **Fehlende Werte (NaN, *Not a Number*):** Wo EB++ für ein Gen in einem Batch keine Verschiebung schätzen konnte (zum Beispiel weil das Gen dort fast überall 0 ist),
  steht kein Wert. Betroffen sind genau die veränderten Proben, also nur bestimmte Tumortypen (Magen, Endometrium, Ovar, Speiseröhre, akute myeloische Leukämie, Dickdarm, Mastdarm, Prostata),
  und vor allem schwach exprimierte Gene, aber auch Markergene wie `KRT5` (Keratin 5) und `SFTPC` (Surfactant-Protein C, Lunge).
- **Negative Werte** zwischen $-1$ und 0: Durch das Verschieben rutschen Werte nahe 0 unter 0.
- Die korrigierten Proben sind **nicht mehr genau normalisiert** (oberes Quartil zwischen etwa 780 und 1.130 statt 1.000).

#### Weitere Probleme

1. **Qualität:** 184 Proben sind von den PanCanAtlas-Autoren als `Do_not_use` („nicht verwenden“) markiert. Einzelne Proben haben auffällig viele Nullen (bis 27 %).
   Bei 15 Proben widerspricht die Expression von `XIST`/`RPS4Y1` dem Geschlecht laut klinischer Tabelle.
2. **Geschlecht als Störfaktor:** Mehrere Tumortypen kommen nur bei einem Geschlecht vor (Prostata, Hoden; Eierstock, Gebärmutterkörper, Karzinosarkom der Gebärmutter, Gebärmutterhals;
   Brustkrebs fast nur bei Frauen).
3. **Ungleiche Gruppengrößen:** Brustkrebs hat rund 30-mal so viele Proben wie Gallengangkrebs.

---

## Teil 2: Entscheidungen für die Aufbereitung

Für jede Entscheidung: die Optionen, was der erste Durchlauf der Hauptkomponentenanalyse vor der Aufbereitung gemacht hat („bisher“) und ein Vorschlag.
**Umgesetzt und geprüft** sind die Vorschläge in `notebook_aufbereitung.ipynb`; die Ergebnisse stehen in der Übersicht am Ende.
Die Reihenfolge entspricht ungefähr der Reihenfolge in der Verarbeitungskette.
Viele Entscheidungen hängen voneinander ab (zum Beispiel entscheidet die Probenauswahl mit darüber, welche Gene fehlende Werte haben).

### A. Welche Proben behalten wir?

**A1. Welche Probentypen?**
- Optionen: nur Primärtumoren (`01`, bei akuter myeloischer Leukämie `03`) · zusätzlich Metastasen · zusätzlich Normalgewebe · zusätzliche Primärtumoren (`05`) und Rezidive (`02`).
- Zu bedenken: Metastasen enthalten Gewebe des Zielorgans (Melanom-Metastasen im Lymphknoten), Normalgewebe bildet eigene Gruppen nahe „seinem“ Tumor.
  Beides verwischt den Vergleich der Tumortypen untereinander. Ohne Metastasen bleiben vom Melanom der Haut nur 103 Proben.
- Bisher: nur `01`/`03`. **Vorschlag:** beibehalten; Normalgewebe gegebenenfalls später als eigene Vergleichsanalyse.

**A2. Eine Probe pro Patient – nach welcher Regel?**
- Betrifft nach A1 nur 4 Patienten mit mehreren Primärtumor-Proben.
- Optionen: erste nach Barcode · niedrigster Code für Gewebeprobe und Teilstück · Mittelwert der Proben.
- Bisher: erste nach Barcode. **Vorschlag:** beibehalten (Auswirkung vernachlässigbar), aber dokumentieren.

**A3. Proben mit `Do_not_use` („nicht verwenden“) entfernen?**
- 184 Proben (155 davon Primärtumoren), zum Beispiel wegen Vorbehandlung, zu wenig Tumorzellen oder falschem Tumortyp.
- Offen: 33 Proben (Ovarialkarzinom) haben keinen Eintrag in der Qualitätsdatei. Behalten oder entfernen?
- Bisher: markierte entfernt, Proben ohne Eintrag behalten. **Vorschlag:** so lassen.

**A4. Proben vom älteren Illumina Genome Analyzer?**
- 598 Proben (Endometriumkarzinom, Dickdarm- und Mastdarmkrebs), dazu 33 Proben ohne Geräteangabe.
- Optionen: entfernen (sauberer, aber das Endometriumkarzinom verliert über die Hälfte, Mastdarmkrebs fast alles) · behalten und der EB++-Korrektur (*Empirical Bayes*) vertrauen ·
  behalten und später prüfen, ob diese Proben eigene Gruppen bilden.
- Bisher: entfernt (nur Illumina HiSeq). Zu prüfen: Wie viel ändert sich, wenn man sie behält?

**A5. Weitere Qualitätsfilter?**
- Kandidaten: Proben mit auffällig vielen Nullen (Schwelle?), die 15 Proben mit Geschlechtswiderspruch, Ausreißer in der Hauptkomponentenanalyse.
- Bisher: nicht gefiltert. **Vorschlag:** erst ansehen, wo diese Proben in der Hauptkomponentenanalyse und in UMAP (*Uniform Manifold Approximation and Projection*) liegen;
  nur mit klarer Begründung entfernen.

**A6. Welche Tumortypen?**
- Optionen: alle 33 · eine thematische Auswahl · Mindestgröße (zum Beispiel mindestens 50 Proben; träfe das diffus großzellige B-Zell-Lymphom und Gallengangkrebs).
- Zusammenlegen? Dick- und Mastdarmkrebs (COAD und READ) werden in der Literatur oft als `COADREAD` zusammengefasst
  (im ersten Durchlauf stammen 68 % der nächsten Nachbarn von Mastdarm-Proben aus dem Dickdarmkrebs). Achtung: Zusammenlegen vorab nimmt ein mögliches Ergebnis vorweg.
- Bisher: alle 33, nichts zusammengelegt. **Vorschlag:** für die Forschungsfrage alle 33 behalten.

**A7. Ungleiche Gruppengrößen ausgleichen?**
- Große Typen prägen die Achsen der Hauptkomponentenanalyse und „ziehen“ Nachbarn an; das Clustering kann kleine Typen übersehen.
- Optionen: nichts tun · Stichprobe mit gleicher Höchstzahl pro Typ ziehen (zum Beispiel höchstens 150) ·
  Hauptkomponentenanalyse auf einer ausgeglichenen Stichprobe berechnen und alle Proben darauf projizieren · Gewichte.
- Bisher: nichts. **Vorschlag:** Hauptanalyse mit allen Proben, als Robustheitsprüfung eine ausgeglichene Stichprobe.

### B. Fehlende und negative Werte

**B1. Wie gehen wir mit den fehlenden Werten (NaN, *Not a Number*) um?**

| Option | Folge |
|---|---|
| Gene mit fehlenden Werten entfernen | bei den Primärtumoren vom Illumina HiSeq 3.820 von 20.531 Genen; darunter Markergene wie `KRT5` (Keratin 5); 1.463 dieser Gene wären sonst unter den 5.000 Genen mit der größten Varianz |
| Proben mit fehlenden Werten entfernen | 1.058 von 9.104 Proben; **Magenkrebs, Speiseröhrenkrebs und akute myeloische Leukämie fallen komplett weg**, das Ovarialkarzinom fast komplett |
| mit 0 auffüllen | einfach; aber ein fehlender Wert heißt nicht „nicht exprimiert“ und betrifft nur bestimmte Tumortypen → könnte künstliche Unterschiede erzeugen |
| mit dem Mittelwert pro Gen auffüllen (über Proben ohne fehlende Werte) | neutral für die Hauptkomponentenanalyse, zieht die betroffenen Tumortypen aber zur Mitte |
| Gene nur entfernen, wenn mehr als x % fehlen, Rest auffüllen | Kompromiss; Schwelle zu wählen |
| Auffüllen über die k nächsten Nachbarn (*k-nearest-neighbors imputation*) | aufwendig, Ergebnis schwer nachzuvollziehen |

- Wichtig: Weil die fehlenden Werte tumortypspezifisch sind, kann jede Auffüllung den Tumortyp „einschreiben“. Gene entfernen ist die vorsichtigste Wahl.
- Bisher: Gene mit fehlenden Werten entfernt. **Vorschlag:** beibehalten; prüfen, wie viele der entfernten Gene nach dem Expressionsfilter (D1) überhaupt noch relevant wären.

**B2. Negative Werte?**
- Sie liegen zwischen $-1$ und $0$ und bedeuten „nicht exprimiert“. Außerdem ist $\log_2(x+1)$ für $x \leq -1$ nicht definiert.
- Bisher: auf 0 gesetzt. **Vorschlag:** beibehalten.

### C. Transformation

**C1. Log-Transformation?**
- Optionen: $\log_2(x+1)$ (Standard) · anderer Pseudocount, also eine andere kleine Zahl statt der 1 (zum Beispiel $+0{,}5$ oder $+0{,}1$; betont Unterschiede bei kleinen Werten stärker) ·
  Areasinus hyperbolicus ($\operatorname{arsinh}$) · keine.
- Ohne Transformation bestimmen wenige extrem hohe Gene jede Varianz- und Abstandsrechnung.
- Bisher: $\log_2(x+1)$. **Vorschlag:** beibehalten.

**C2. Reihenfolge von Filtern und Transformation**
- Expressions- und Varianzfilter (D1, D2) auf der **logarithmierten Skala** berechnen, sonst gewinnen bei der Varianz immer die hoch exprimierten Gene.
- Zentrieren und Standardisieren erst **nach** der Probenauswahl, damit die Mittelwerte zur verwendeten Stichprobe passen.

**C3. Zentrieren** ist für die Hauptkomponentenanalyse nötig (Mittelwert pro Gen abziehen). Keine echte Entscheidung, aber: Mittelwert über welche Proben
(alle oder eine ausgeglichene Stichprobe, siehe A7)?

**C4. Pro Gen standardisieren (z-Werte, also zusätzlich durch die Standardabweichung teilen)?**
- Ohne: Gene mit großer Streuung (gewebespezifische Gene) dominieren die Achsen. Das ist für die Frage nach Tumortypen eher erwünscht.
- Mit: jedes Gen zählt gleich, auch schwach streuende, verrauschte Gene; setzt einen vorherigen Filter (D1, D2) voraus, sonst wird Rauschen verstärkt.
- Bisher: nur zentriert. **Vorschlag:** nur zentrieren; Standardisierung als Vergleich.

**C5. Proben erneut normalisieren?**
- Durch die Batch-Korrektur liegt das obere Quartil nicht mehr bei allen Proben genau bei 1.000 (Spanne etwa 780 bis 1.130).
- Optionen: nicht neu normalisieren · Proben erneut auf gleiches oberes Quartil oder gleichen Median skalieren.
- **Vorschlag:** nicht neu normalisieren (würde die Batch-Korrektur teilweise wieder aufheben).

### D. Welche Gene behalten wir?

**D1. Schwach exprimierte Gene entfernen?**
- Sie tragen wenig Information, aber viel Rauschen (siehe 1.5: kleine Werte sind ungenau); nach Standardisierung (C4) wird das Rauschen sogar verstärkt.
- Mögliche Kriterien: Gen in mindestens x % der Proben größer als 0 · mittlere log-Expression mindestens 1 · in mindestens einem Tumortyp ausreichend exprimiert
  (wichtig, damit Markergene kleiner Tumortypen nicht wegfallen).
- Bisher: kein Filter (außer fehlenden Werten). **Vorschlag:** einfaches Kriterium, zum Beispiel „in mindestens einem Tumortyp bei mindestens der Hälfte der Proben größer als 0“;
  konstante Gene (Varianz 0) immer entfernen.

**D2. Nur die $k$ Gene mit der größten Varianz?**
- Übliche Werte: $k$ = 1.000 bis 5.000. Verringert Rechenaufwand und Rauschen; die Streuung fällt nach den ersten paar Tausend Genen deutlich ab.
- Zu bedenken: Varianz über alle Proben bevorzugt Gene, die große Tumortypen trennen. Alternative: Varianz in einer ausgeglichenen Stichprobe (A7).
- Bisher: alle 16.711 verbleibenden Gene. **Vorschlag:** zum Beispiel $k$ = 5.000 als Standard, $k$ = 1.000 und „alle“ als Robustheitsprüfung.

**D3. Geschlechtsgene entfernen?**
- `XIST` (*X-inactive specific transcript*), `RPS4Y1` (ribosomales Protein S4 vom Y-Chromosom) und weitere Gene auf X- und Y-Chromosom streuen stark
  und können eine eigene Achse bilden, die nur das Geschlecht abbildet.
- Optionen: behalten · nur die Gene des Y-Chromosoms und `XIST` entfernen · alle Gene auf X- und Y-Chromosom entfernen.
- Problem: Die Datei enthält keine Zuordnung zu Chromosomen; dafür bräuchten wir eine zusätzliche Tabelle, zum Beispiel vom National Center for Biotechnology Information
  oder vom HUGO Gene Nomenclature Committee (Ausschuss für Gennamen der Human Genome Organisation), verknüpft über die Entrez-Nummer.
- Bisher: nichts entfernt. Die Gene des Y-Chromosoms und `XIST` sind noch in den Daten (anders als zunächst angenommen, fallen sie nicht über den Filter für fehlende Werte weg).
  **Vorschlag:** zuerst prüfen, ob in den ersten Hauptkomponenten eine Geschlechtsachse auftaucht; wenn ja, Geschlechtsgene entfernen.

**D4. Namen der Gene:** 29 Gene ohne Symbol und das doppelte `SLC35E2` bekommen die Entrez-Nummer angehängt (schon umgesetzt). Keine weitere Entscheidung nötig.

### E. Batch-Effekte und Störfaktoren

**E1. Zusätzliche Batch-Korrektur (zum Beispiel nach Platte)?**
- Platten und Sequenzierzentren sind stark mit dem Tumortyp vermengt. Eine Korrektur nach Platte würde echte Unterschiede zwischen Tumortypen mit entfernen.
- **Vorschlag:** keine weitere Korrektur, stattdessen kontrollieren (E2).

**E2. Welche Störgrößen prüfen wir?**
- Ergebnisse (Hauptkomponenten, UMAP-Darstellung, Cluster) einfärben nach: Platte, Sequenzierzentrum, Sequenziergerät (falls Proben vom Illumina Genome Analyzer behalten werden),
  Geschlecht, Alter, Anteil Nullen.
- Am aussagekräftigsten **innerhalb** eines großen Tumortyps (zum Beispiel Brustkrebs nach Platte), weil dort der Tumortyp als Erklärung wegfällt.

**E3. Tumorreinheit (*Purity*)?**
- Proben enthalten neben Tumorzellen Immun- und Bindegewebszellen (siehe 1.5; die vierte Hauptkomponente ist eine Immunzell-Achse). Das kann Gruppen aufspalten, ohne dass es am Tumortyp liegt.
- Eine Schätzung der Reinheit (Verfahren ABSOLUTE) gibt es auf der PanCanAtlas-Seite als eigene Datei. **Offen:** dazuladen, um Aufspaltungen zu erklären?

### F. Aufbereitung für die Methoden

**F1. Wie viele Hauptkomponenten als Eingabe für UMAP und Clustering?**
- Kein klarer Knick; nach 15 bis 20 Hauptkomponenten flacht die Kurve ab, 50 Hauptkomponenten erklären etwa 70 % der Varianz.
- Bisher: 50. **Vorschlag:** 50 als Standard, 20 und 100 als Vergleich.

**F2. Abstandsmaß**
- Euklidischer Abstand auf den Hauptkomponenten (Standard) oder Korrelations- beziehungsweise Kosinusabstand
  (in der Genexpression verbreitet; unempfindlich gegen Unterschiede im Gesamtniveau einer Probe).

**F3. Rechenaufwand beim spektralen Clustering**
- Bei $n \approx 9.000$ Proben ist eine volle $n \times n$-Ähnlichkeitsmatrix (etwa 320 Megabyte als 32-Bit-Gleitkommazahlen) noch machbar;
  ein dünn besetzter Graph der k nächsten Nachbarn ist aber schneller.
- Zu wählen: Größe der Nachbarschaft $k$ und Bandbreite der Ähnlichkeitsfunktion.

**F4. Reproduzierbarkeit**
- Aufbereitung als eine Funktion mit allen Parametern (Probenfilter, Regel für fehlende Werte, Anzahl Gene $k$, Standardisierung und so weiter) schreiben und die fertige Matrix speichern,
  damit alle Notebooks dieselben Daten verwenden und Robustheitsprüfungen nur Parameter ändern.
- Zufallsstartwerte (*seeds*) für Stichproben, die abgeschnittene Singulärwertzerlegung, UMAP und Clustering festlegen.

---

## Übersicht

Spalte „Ergebnis“: Umsetzung und Prüfung in `notebook_aufbereitung.ipynb`.

| Nummer | Entscheidung | erster Durchlauf der Hauptkomponentenanalyse (vor der Aufbereitung) | Vorschlag | Ergebnis (`notebook_aufbereitung.ipynb`) |
|---|---|---|---|---|
| A1 | Probentypen | nur Primärtumoren `01`/`03` | beibehalten | umgesetzt |
| A2 | eine Probe pro Patient | erste nach Barcode | beibehalten | umgesetzt (betrifft 3 Proben) |
| A3 | `Do_not_use` („nicht verwenden“) | entfernt | beibehalten | umgesetzt |
| A4 | Proben vom Illumina Genome Analyzer | entfernt | Auswirkung prüfen | Proben vom älteren Gerät liegen bevorzugt neben ihresgleichen → **entfernt** |
| A5 | weitere Qualitätsfilter | keine | erst ansehen | Proben mit vielen Nullen oder Geschlechtswiderspruch liegen bei ihrem Tumortyp → **kein Filter** |
| A6 | Tumortypen | alle 33 | beibehalten | umgesetzt |
| A7 | Gruppengrößen | nicht ausgeglichen | ausgeglichene Stichprobe als Vergleich | ändert nichts → **kein Ausgleich** |
| B1 | fehlende Werte (NaN) | Gene entfernt | beibehalten, nach D1 erneut prüfen | 1.463 entfernte Gene wären wichtig (Keratine), Auffüllen mit 0 ändert das Ergebnis aber nicht → **Gene entfernen** |
| B2 | negative Werte | auf 0 gesetzt | beibehalten | umgesetzt |
| C1 | Log-Transformation | $\log_2(x+1)$ | beibehalten | umgesetzt |
| C4 | Standardisieren | nein | nein, als Vergleich ja | nein; Vergleich noch offen |
| C5 | Proben neu normalisieren | nein | nein | nein |
| D1 | Expressionsfilter | keiner | einführen | umgesetzt; entfernt nach B1 kaum noch Gene |
| D2 | $k$ Gene mit größter Varianz | alle | $k$ = 5.000, Vergleich 1.000 und alle | $k$ = 5.000; Vergleich noch offen |
| D3 | Geschlechtsgene | nichts entfernt | prüfen, gegebenenfalls entfernen | Geschlechtsachse auf Hauptkomponente 9 → **16 Gene entfernt** (aus den Daten bestimmt, ohne Chromosomentabelle) |
| E1 | zusätzliche Batch-Korrektur | keine | keine | keine |
| E2 | Kontrolle der Störgrößen | keine | einfärben nach Platte, Sequenzierzentrum, Geschlecht | kleiner Platteneffekt ab Hauptkomponente 8, deutlich innerhalb des Brustkrebses; kleiner Rest-Zentrumseffekt beim Magenkrebs |
| E3 | Tumorreinheit | nicht verwendet | offen | offen |
| F1 | Anzahl Hauptkomponenten | 50 | 50, Vergleich 20 und 100 | 100 gespeichert; Vergleich noch offen |
| F2 | Abstandsmaß | euklidisch | euklidisch, gegebenenfalls Korrelation als Vergleich | offen |
