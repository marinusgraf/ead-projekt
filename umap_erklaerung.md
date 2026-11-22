# UMAP: Wie das Verfahren funktioniert

Dieses Dokument erklärt UMAP (*Uniform Manifold Approximation and Projection*) Schritt für Schritt und bezieht es auf unser Projekt.
Die Berechnungen stehen in `notebook_umap.ipynb`; dort werden die ersten 50 Hauptkomponenten aus `notebook_pca.ipynb` eingebettet (9.720 Primärtumoren).

---

## 1. Die Idee

UMAP sucht eine zweidimensionale Anordnung der Proben, in der jede Probe **dieselben Nachbarn** hat wie im hochdimensionalen Raum.
Abstände will UMAP nicht erhalten, nur Nachbarschaften.

| | Hauptkomponentenanalyse | UMAP |
|---|---|---|
| Ziel | möglichst viel **Varianz** in wenigen Achsen | möglichst viele **Nachbarschaften** in 2 Dimensionen |
| Abbildung | linear (Drehung und Projektion) | nichtlinear (Punkte werden frei verschoben) |
| Achsen | haben eine Bedeutung (Ladungen auf Gene) | haben keine Bedeutung |
| Ergebnis | eindeutig | hängt von Zufall und Parametern ab |

Das Verfahren hat zwei Schritte:

1. Aus den Daten wird ein **gewichteter Nachbarschaftsgraph** gebaut (Abschnitt 2).
2. Eine Anordnung in der Ebene wird so verschoben, dass ihr eigener Nachbarschaftsgraph dem ersten möglichst gleicht (Abschnitte 3 und 4).

---

## 2. Schritt 1: Der Nachbarschaftsgraph

### 2.1 Abstände

Jede Probe ist ein Punkt im $\mathbb{R}^{50}$: ihre Koordinaten (Scores $Z = U S$) auf den ersten 50 Hauptkomponenten.
Abstände sind **euklidisch** in diesem Raum.

- Die Hauptkomponentenanalyse ist eine Drehung der zentrierten Genmatrix. Der Abstand im $\mathbb{R}^{50}$ ist daher der euklidische Abstand der 9.089 $\log_2$-Expressionswerte,
  nachdem man sie auf die ersten 50 Achsen projiziert hat. Die Einheit bleibt $\log_2$-Expression.
- Die Scores sind **nicht** pro Komponente standardisiert. Hauptkomponente 1 hat die größte Streuung und trägt am meisten zu den Abständen bei, Hauptkomponente 50 nur wenig.
  Würde man jede Komponente auf Varianz 1 bringen („Weißen“), zählten auch die verrauschten hinteren Achsen voll mit.
- Andere Abstandsmaße wären möglich (Parameter `metric`, zum Beispiel Kosinus- oder Korrelationsabstand).
  Wir bleiben beim euklidischen Abstand, damit UMAP zur Hauptkomponentenanalyse und zur Nachbarschaftsanalyse passt.

**Warum die Hauptkomponentenanalyse vorher?** UMAP könnte auch direkt auf den 9.089 Genen laufen. Die 50 Hauptkomponenten entfernen aber Rauschen
(in sehr vielen Dimensionen werden alle Abstände ähnlich) und machen die Nachbarsuche viel schneller.
Das Ergebnis hängt kaum davon ab: Der Anteil der Nachbarn vom gleichen Tumortyp liegt bei 84,4 % (50 Hauptkomponenten) gegenüber 85,6 % (alle 9.089 Gene),
siehe Abschnitt 5.2 im UMAP-Notebook.

### 2.2 Die k nächsten Nachbarn

Für jede Probe werden die $k$ nächsten Nachbarn gesucht, bei uns `n_neighbors=15`.

- UMAP zählt die Probe selbst mit: Jede Probe hat also **14 echte Nachbarn**.
- Ab etwa 4.000 Punkten berechnet `umap-learn` nicht mehr alle Abstände, sondern sucht die Nachbarn **näherungsweise** (Verfahren NN-Descent aus dem Paket `pynndescent`, mit Zufallsanteil).
  Bei unseren 9.720 Proben ist das der Fall. Die exakten Nachbarn berechnen wir im UMAP-Notebook (Abschnitt 2) separat durch Vergleich aller Paare.

Die Nachbarsuche legt fest, **welche** Kanten es gibt. Die Zahl ist für alle Proben gleich.

### 2.3 Gewichte: der lokale Maßstab $\rho_i$ und $\sigma_i$

Jede Kante bekommt ein Gewicht zwischen 0 und 1. Dafür hat jede Probe $i$ ihren **eigenen Maßstab**:

$$
w_{i \to j} = \exp\!\left(-\frac{d_{ij} - \rho_i}{\sigma_i}\right)
$$

- $\rho_i$ ist der Abstand zum allernächsten Nachbarn. Dieser bekommt immer Gewicht 1.
- $\sigma_i$ wird für jede Probe so eingestellt, dass die **Summe der Gewichte** ihrer Nachbarn immer gleich ist:

$$
\sum_{j} w_{i \to j} = \log_2(k) \approx 3{,}9 \quad \text{für } k = 15
$$

Die **Zahl** der Nachbarn ist also ohnehin fest. $\sigma_i$ regelt, wie **stark** sie verbunden sind, und zwar relativ zur Dichte der Umgebung.

**Beispiel:**

| | Probe A in dichter Gruppe (BRCA, 1.083 Proben) | Probe B in dünner Gruppe (CHOL, 36 Proben) |
|---|---|---|
| Abstände der Nachbarn | 10; 10,5; 11; …; 14 | 30; 35; 40; …; 80 |
| $\sigma_i$ | klein | groß |
| Zusammenhalt mit der Umgebung | gleich stark | gleich stark |

Mit einem gemeinsamen Maßstab für alle Proben wären die Kanten von B fast alle schwach, B wäre im Graphen kaum verbunden.
Durch den eigenen Maßstab gleicht UMAP Dichteunterschiede bewusst aus (daher „Uniform“ im Namen).
**Folge:** Absolute Abstände und Dichten aus dem $\mathbb{R}^{50}$ gehen verloren.

### 2.4 Symmetrisieren

Die Gewichte sind zunächst **gerichtet**: A kann B als Nachbarn haben, ohne dass B auch A als Nachbarn hat.
UMAP verbindet beide Richtungen mit

$$
w_{ij} = a + b - a \cdot b, \qquad a = w_{i \to j},\; b = w_{j \to i}.
$$

Das ist die Wahrscheinlichkeit, dass mindestens eine Seite die Kante sieht (eine „unscharfe Vereinigung“).

**Folge: Ein Knoten kann mehr als 14 Kanten haben.**

- **Ausgehende** Kanten (die eigenen Nachbarn): für jede Probe genau 14.
- **Eingehende** Kanten (wie oft die Probe selbst als Nachbar gewählt wird): nicht festgelegt, von 0 bis zu einigen Dutzend.
- Nach dem Symmetrisieren zählen beide. Eine Probe, die selbst 14 Nachbarn wählt und zusätzlich von 3 anderen gewählt wird, hat 17 Kanten.
  Die 3 zusätzlichen Kanten sind schwächer ($b = 0$, also $w_{ij} = a$).
- Proben **in der Mitte einer Gruppe** sammeln viele eingehende Kanten („Hubs“). Proben **am Rand** oder Ausreißer behalten etwa ihre 14.
- **Gegenseitige** Nachbarschaften zählen nur einmal und bekommen ein hohes Gewicht.

Ergebnis von Schritt 1: ein symmetrischer, dünn besetzter, gewichteter Graph mit der Gewichtsmatrix $W$ ($n \times n$).

---

## 3. Der Start: spektrale Einbettung des Graphen

Bevor die Punkte verschoben werden, braucht UMAP eine Startanordnung. Es nimmt die **spektrale Einbettung** des Graphen.

### 3.1 Bausteine

- $W$: Gewichtsmatrix aus Schritt 1, $w_{ij}$ ist das Kantengewicht (0 ohne Kante).
- $D$: Gradmatrix, diagonal mit $d_i = \sum_j w_{ij}$ (Summe der Kantengewichte von Probe $i$).
- $L = D - W$: die **Laplace-Matrix** des Graphen.

### 3.2 Die Aufgabe: verbundene Proben nah beieinander

Gesucht ist zunächst **eine** Koordinate pro Probe, also ein Vektor $y = (y_1, \dots, y_n)$; $y_i$ ist die Position von Probe $i$ auf einer Linie.
Gut ist eine Anordnung, wenn stark verbundene Proben nah beieinander liegen. Als Strafe dient

$$
E(y) = \frac{1}{2} \sum_{i,j} w_{ij} \, (y_i - y_j)^2 = y^\top L \, y .
$$

Starke Kanten zwischen weit entfernten Punkten kosten viel, fehlende Kanten kosten nichts.

### 3.3 Die beiden Nebenbedingungen

$E(y)$ hängt nur von den **Differenzen** $y_i - y_j$ ab. Daraus folgen zwei Freiheiten, die eine einfache Minimierung ausnutzen würde:

1. **Verschieben ändert nichts:** Addiert man zu allen $y_i$ dieselbe Zahl $c$, bleibt $E$ gleich.
   Im Extremfall $y = (c, c, \dots, c)$ haben alle Proben dieselbe Koordinate, und $E = 0$. Diese Anordnung unterscheidet keine Probe von einer anderen.
2. **Stauchen verkleinert $E$:** Multipliziert man alle $y_i$ mit $s$, wird $E$ um den Faktor $s^2$ kleiner, obwohl sich die Reihenfolge der Proben nicht ändert.
   Im Extremfall $y = (0, \dots, 0)$ ist $E = 0$.

Die Nebenbedingungen legen diese beiden Freiheiten fest:

| Freiheit von $E$ | nutzlose Lösung | Nebenbedingung | gewichtete Form (UMAP) |
|---|---|---|---|
| Verschieben | $y = (c, \dots, c)$ | Mittelwert 0: $\sum_i y_i = 0$ | $\sum_i d_i \, y_i = 0$ |
| Stauchen | $y = (0, \dots, 0)$ | Länge 1: $\sum_i y_i^2 = 1$ | $\sum_i d_i \, y_i^2 = y^\top D y = 1$ |

Die erste Bedingung heißt auch „senkrecht zum Vektor $(1, \dots, 1)$“. Unter beiden Bedingungen kann $E$ nur klein werden, wenn die Anordnung tatsächlich gut ist.

**Zahlenbeispiel:** drei Proben, Kanten A–B mit Gewicht 1 und B–C mit Gewicht 0,1 (ungewichtete Bedingungen).

| $y = (y_A, y_B, y_C)$ | Mittelwert 0? | Länge 1? | $E(y)$ |
|---|---|---|---|
| $(5;\ 5;\ 5)$ | nein | nein | 0 |
| $(0;\ 0;\ 0)$ | ja | nein | 0 |
| $(-0{,}71;\ 0;\ 0{,}71)$ | ja | ja | $0{,}5 + 0{,}05 = 0{,}55$ |
| $(-0{,}44;\ -0{,}38;\ 0{,}82)$ | ja | ja | $0{,}004 + 0{,}142 \approx 0{,}146$ |

Die letzte Zeile ist die beste mögliche Anordnung: A und B liegen dicht beieinander (starke Kante), C ist abgesetzt (schwache Kante).

### 3.4 Die Lösung: Eigenvektoren

Unter diesen Nebenbedingungen wird $y^\top L y$ von einem **Eigenvektor zum kleinsten Eigenwert** minimiert (Rayleigh-Quotient).
Mit den gewichteten Bedingungen ist es das verallgemeinerte Eigenwertproblem $L y = \lambda D y$, also die normalisierte Laplace-Matrix.

- Eigenwert $\lambda_1 = 0$: Der Eigenvektor ist konstant. Er verletzt die erste Nebenbedingung und wird verworfen.
- Eigenwert $\lambda_2$: Sein Eigenvektor ist die beste Anordnung auf einer Linie, der **Fiedler-Vektor**. Der Wert von $E$ ist genau $\lambda_2$
  (im Zahlenbeispiel $0{,}146$).
- Eigenwert $\lambda_3$: Sein Eigenvektor ist die beste weitere Koordinate, senkrecht zur vorigen.

Für die Ebene nimmt man **die Eigenvektoren 2 und 3** als $x$- und $y$-Koordinate jeder Probe.

**Anschaulich:** Bei zwei Gruppen, die innen dicht und untereinander nur schwach verbunden sind, ist der Fiedler-Vektor in der einen Gruppe positiv und in der anderen negativ.
Er legt den Schnitt dorthin, wo am wenigsten Kantengewicht durchtrennt wird.
Zerfällt der Graph in $c$ getrennte Teile, hat $L$ genau $c$ Eigenwerte gleich 0. Fast getrennte Teile zeigen sich als kleine Eigenwerte nahe 0.

### 3.5 Die Rolle in UMAP

1. UMAP berechnet die Eigenvektoren 2 und 3 mit einem Löser für dünn besetzte Matrizen (`eigsh` aus SciPy).
2. Die Koordinaten werden auf etwa $-10$ bis $10$ skaliert und leicht verrauscht.
3. Zerfällt der Graph in mehrere Teile, wird jeder Teil einzeln eingebettet, und die Teile werden nebeneinander angeordnet.

Der Start enthält schon die **globale Struktur** (welche großen Gruppen zusammenhängen). Der folgende Schritt verschiebt vor allem lokal.
Deshalb erhält UMAP die grobe Anordnung besser als t-SNE (*t-distributed Stochastic Neighbor Embedding*), das traditionell zufällig startet.

---

## 4. Schritt 2: Die Anordnung optimieren

In der Ebene misst UMAP die Ähnlichkeit zweier Punkte mit Abstand $d$ (euklidisch im $\mathbb{R}^2$) als

$$
v_{ij} = \frac{1}{1 + a \, d_{ij}^{2b}} ,
$$

wobei $a$ und $b$ aus `min_dist` bestimmt werden. Bis etwa zum Abstand `min_dist` gilt ein Paar als voll ähnlich.

Optimiert wird die **Kreuzentropie** zwischen den Gewichten $w_{ij}$ (hochdimensional) und $v_{ij}$ (Ebene), mit stochastischem Gradientenabstieg:

- **Anziehung:** Paare mit Kante im Graphen werden zueinander gezogen, stärker bei großem $w_{ij}$.
- **Abstoßung:** Zufällig gezogene Paare werden auseinandergeschoben (**Negativstichproben**).
  Weil nicht alle $n^2$ Paare betrachtet werden müssen, ist UMAP schnell.

Optimiert wird also, dass zwei **Graphen** zueinander passen, nicht die Abstände selbst.

---

## 5. Die Parameter

| Parameter | bei uns | Wirkung |
|---|---|---|
| `n_neighbors` | 15 | klein: nur die engste Umgebung zählt, viele kleine Inseln. Groß: mehr globale Struktur, gröbere Gruppen |
| `min_dist` | 0,5 | wie dicht Punkte zusammenrücken dürfen. Klein: kompakte Klumpen. Groß: lockerere, besser lesbare Wolken |
| `metric` | euklidisch | Abstandsmaß im Eingangsraum |
| `random_state` | 0 | Startwert des Zufalls (Nachbarsuche, Rauschen, Negativstichproben). Ein anderer Wert ergibt ein anderes Bild |

Im UMAP-Notebook (Abschnitt 5) sind `n_neighbors` $\in \{5, 15, 50\}$, `min_dist` $\in \{0{,}1;\ 0{,}5\}$, drei Eingaben und drei Startwerte verglichen.
`min_dist = 0,5` haben wir wegen der Lesbarkeit gewählt; die Kennzahlen ändern sich dadurch kaum.

---

## 6. Was man aus einem UMAP-Bild ablesen darf

**Verlässlich:**

- welche Proben **lokal zusammengehören**, zum Beispiel welche Tumortypen sich mischen,
- dass eine Gruppe in **Teilgruppen** zerfällt (bei uns Speiseröhrenkrebs in Plattenepithel- und Adenokarzinom, Brustkrebs nach Östrogenrezeptor).

**Nicht verlässlich:**

- **Abstände zwischen Inseln:** Zwei weit entfernte Inseln sind nicht unbedingt unähnlicher als zwei nahe.
- **Größe und Dichte einer Insel:** UMAP gleicht Dichten bewusst aus (Abschnitt 2.3).
- **Achsen:** Es gibt keine „UMAP-Achse 1“ mit Bedeutung.

**Unsere Zahlen** zeigen diese Grenzen:

| Kennzahl | Wert |
|---|---|
| Nachbarn aus dem $\mathbb{R}^{50}$, die auch in der Ebene Nachbarn sind | 33,1 % |
| gleiche Nachbarn zwischen den Startwerten 0, 1 und 2 | 57–62 % |
| Anteil der Nachbarn vom gleichen Tumortyp im $\mathbb{R}^{50}$ | 84,4 % |
| Anteil der Nachbarn vom gleichen Tumortyp in der Ebene | 82,0 % |

Die einzelnen Nachbarschaften sind also unsicher, die grobe Aussage (Proben liegen bei ihrem Tumortyp) bleibt stabil.
Deshalb rechnen wir die Nachbarschaftsanalyse auf den 50 Hauptkomponenten und nutzen UMAP vor allem als Bild.

---

## 7. Bezug zur Clusteranalyse

Die **spektrale Clusteranalyse** ist dieselbe Rechnung wie der Start von UMAP, nur mit einem anderen letzten Schritt:

| | spektrale Einbettung (UMAP-Start) | spektrale Clusteranalyse |
|---|---|---|
| Graph | Nachbarschaftsgraph | Nachbarschaftsgraph (oder Ähnlichkeitsgraph) |
| Eigenvektoren | 2 (Nummer 2 und 3) | $k$ für $k$ Cluster |
| danach | Gradientenabstieg | k-Means auf den Eigenvektor-Koordinaten |

Die Eigenvektoren zu kleinen Eigenwerten sind innerhalb der Gruppen des Graphen nahezu konstant, deshalb trennt k-Means sie leicht.

**Vorsicht beim Clustern auf UMAP-Koordinaten:** Die klaren Inseln entstehen teilweise erst durch UMAP selbst (Abstoßung, `min_dist`).
Für die Aufteilung einzelner Tumortypen (Abschnitt 3.1 im UMAP-Notebook) ist k-Means in der Ebene vertretbar, weil wir das Ergebnis gegen den histologischen Typ geprüft haben.
Für die eigentliche Clusteranalyse sind die 50 Hauptkomponenten oder der Nachbarschaftsgraph die sauberere Grundlage.
