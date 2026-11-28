"""Abbildungen für die Zwischenpräsentation.

Aufruf aus dem Projektordner:  .venv/bin/python praesentation/abbildungen.py
Voraussetzung: notebook_aufbereitung.ipynb, notebook_pca.ipynb und notebook_umap.ipynb wurden einmal ausgeführt
(Binärdatei der Rohmatrix, aufbereitete Matrix, Hauptkomponenten, UMAP-Einbettung).
Die Rechnungen folgen den Notebooks; die zusätzlichen UMAP-Varianten werden beim ersten Lauf in
data/pancanatlas/praesentation_umap_varianten_gene.npz zwischengespeichert.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.sparse.linalg import svds
from scipy.cluster.vq import kmeans2

warnings.filterwarnings("ignore", message="IProgress not found")
warnings.filterwarnings("ignore", message="n_jobs value")

DATA_DIR = Path("data/pancanatlas")
OUT = Path("praesentation/abbildungen")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 0

# ---------------------------------------------------------------- Schrift und Farben passend zu den Folien
for f in ["FiraSans-Regular.otf", "FiraSans-SemiBold.otf"]:
    for ordner in ["/usr/share/texmf-dist", "/usr/share/texlive/texmf-dist"]:     # Arch bzw. Debian/Ubuntu
        p = Path(ordner) / "fonts/opentype/public/fira" / f
        if p.exists():
            font_manager.fontManager.addfont(str(p))
            break
TEXT, MUTED, GRID = "#17212b", "#5c6872", "#d9dee2"
BLAU, ORANGE, AQUA, GELB, ROSA, GRAU_SERIE = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#9aa3ab"
HELLGRAU = "#dfe3e6"
plt.rcParams.update({
    "font.family": "Fira Sans", "font.size": 8, "text.color": TEXT,
    "axes.labelcolor": TEXT, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 8.5, "axes.titleweight": "semibold",
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})


def de(x, nachkomma=0):
    """Zahl im deutschen Format: Tausenderpunkt, Dezimalkomma."""
    s = f"{x:,.{nachkomma}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def prozent(x, nachkomma=0):
    return de(100 * x, nachkomma) + " %"


def speichern(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("  gespeichert:", name)


def ohne_achsen(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


# ---------------------------------------------------------------- Tumortypen: Gruppen, Farben, Namen
GRUPPEN = {   # Farbe nach Gewebegruppe; jeder Tumortyp wird zusätzlich direkt beschriftet
    "Plattenepithel, Harnblase": (BLAU, ["HNSC", "LUSC", "CESC", "BLCA"]),
    "Verdauungstrakt, Leber": (ORANGE, ["ESCA", "STAD", "COAD", "READ", "PAAD", "LIHC", "CHOL"]),
    "Niere": (AQUA, ["KIRC", "KIRP", "KICH"]),
    "Gehirn, Nebenniere, Melanome": (GELB, ["GBM", "LGG", "PCPG", "ACC", "UVM", "SKCM"]),
    "Brust, Gebärmutter, Eierstock": (ROSA, ["BRCA", "UCEC", "UCS", "OV"]),
    "weitere": (GRAU_SERIE, ["LUAD", "MESO", "THCA", "PRAD", "TGCT", "SARC", "LAML", "DLBC", "THYM"]),
}
FARBE = {t: farbe for farbe, typen in GRUPPEN.values() for t in typen}
assert len(FARBE) == 33


def legende_gruppen(ax, **kw):
    h = [plt.Line2D([], [], marker="o", ls="", ms=5, color=f, label=g) for g, (f, _) in GRUPPEN.items()]
    ax.legend(handles=h, handletextpad=0.2, **kw)


def streubild(ax, E, study, s=2.0, beschriften=True, alpha=0.55, label_size=6.2, nur=None):
    # graue Gruppe zuerst, damit die farbigen Gruppen obenauf liegen
    reihenfolge = list(GRUPPEN)[::-1]
    for g in reihenfolge:
        farbe, typen = GRUPPEN[g]
        m = np.isin(study, typen)
        ax.scatter(E[m, 0], E[m, 1], s=s, color=farbe, alpha=alpha, linewidths=0, rasterized=True)
    if beschriften:
        platziert = []
        spanne = np.ptp(E, axis=0)
        for t in pd.Series(study).value_counts().index:     # große Typen zuerst
            if nur is not None and t not in nur:
                continue
            x, y = np.median(E[study == t], axis=0)
            # einfache Kollisionsvermeidung: Beschriftung nach oben/unten schieben, bis sie frei ist
            for versuch in [0, 1, -1, 2, -2, 3, -3]:
                yy = y + versuch * 0.045 * spanne[1]
                if all(abs(x - px) > 0.075 * spanne[0] or abs(yy - py) > 0.04 * spanne[1] for px, py in platziert):
                    break
            platziert.append((x, yy))
            ax.annotate(t, (x, yy), fontsize=label_size, ha="center", va="center", fontweight="semibold", color=TEXT,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))


# ================================================================ Daten laden
print("Lade Daten ...")
X_all = np.load(DATA_DIR / "geneExp_samples_x_genes.npy")                 # 11.069 × 20.531, Rohwerte
barcode = pd.read_csv(DATA_DIR / "geneExp_samples.csv")["barcode"]
gene_id = pd.read_csv(DATA_DIR / "geneExp_genes.csv")["gene_id"]
symbol = gene_id.str.split("|").str[0].to_numpy()

teile = barcode.str.split("-", expand=True)
cdr = pd.read_excel(DATA_DIR / "TCGA-CDR-SupplementalTableS1.xlsx", sheet_name="TCGA-CDR").set_index("bcr_patient_barcode")
quality = pd.read_csv(DATA_DIR / "merged_sample_quality_annotations.tsv", sep="\t")
rna_q = (quality[quality["platform"].isin(["IlluminaHiSeq_RNASeqV2", "IlluminaGA_RNASeqV2"])]
         .sort_values("platform", ascending=False).drop_duplicates("aliquot_barcode")
         .set_index("aliquot_barcode").reindex(barcode))
meta_all = pd.DataFrame({
    "barcode": barcode, "patient": teile[0] + "-" + teile[1] + "-" + teile[2], "sample_type": teile[3].str[:2],
})
meta_all["study"] = meta_all["patient"].map(cdr["type"]).fillna(rna_q["cancer type"].reset_index(drop=True))
meta_all["platform"] = rna_q["platform"].map({"IlluminaHiSeq_RNASeqV2": "HiSeq", "IlluminaGA_RNASeqV2": "GA"}).fillna("ohne Angabe").to_numpy()
meta_all["do_not_use"] = rna_q["Do_not_use"].to_numpy()
meta_all["gender"] = meta_all["patient"].map(cdr["gender"]).map({"FEMALE": "weiblich", "MALE": "männlich"})
assert meta_all["study"].nunique() == 33

# aufbereitete Daten (Standardlauf) aus den Notebooks
meta = pd.read_csv(DATA_DIR / "aufbereitet_proben.csv", dtype={"sample_type": str, "center": str, "portion": str})
genes = pd.read_csv(DATA_DIR / "aufbereitet_gene.csv")
Xc = np.load(DATA_DIR / "aufbereitet_matrix.npy")
Z = np.load(DATA_DIR / "pca_scores.npy")
Vt = np.load(DATA_DIR / "pca_ladungen.npy")
ev = np.load(DATA_DIR / "pca_erklaerte_varianz.npy")
E = np.load(DATA_DIR / "umap_2d.npy")
nb_ein = np.load(DATA_DIR / "nachbarn_eingabe.npy")                     # 15 nächste Nachbarn in der UMAP-Eingabe (alle Gene)
study = meta["study"].to_numpy()
zeile = pd.Series(np.arange(len(meta_all)), index=meta_all["barcode"]).loc[meta["barcode"]].to_numpy()   # Zeilen in X_all
import json
PARAMETER = json.load(open(DATA_DIR / "aufbereitet_parameter.json"))
GESCHLECHTSGENE = PARAMETER["parameter"]["entferne_gene"]          # im Aufbereitungs-Notebook aus den Daten bestimmt


def pca(Xc, k=50):
    U, S, V = svds(Xc.astype(np.float64), k=k, random_state=SEED)
    o = np.argsort(S)[::-1]
    U, S, V = U[:, o], S[o], V[o]
    gesamt = (Xc.astype(np.float64) ** 2).sum() / (Xc.shape[0] - 1)
    return U * S, V, S**2 / (Xc.shape[0] - 1) / gesamt


def nachbarn(Y, K=15):
    Y = np.asarray(Y, dtype=np.float32)
    sq = (Y**2).sum(axis=1)
    nb = np.empty((len(Y), K), dtype=np.int64)
    for s in range(0, len(Y), 2000):
        d2 = sq[s:s + 2000, None] - 2 * Y[s:s + 2000] @ Y.T + sq[None, :]
        d2[np.arange(d2.shape[0]), np.arange(s, s + d2.shape[0])] = np.inf
        nb[s:s + 2000] = np.argpartition(d2, K, axis=1)[:, :K]
    return nb


def eta2(y, g):
    g = pd.Series(np.asarray(g, dtype=object))
    ok = g.notna().to_numpy()
    y, g = np.asarray(y)[ok], g[ok].to_numpy()
    mittel = pd.Series(y).groupby(g).transform("mean").to_numpy()
    return ((mittel - y.mean()) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def innerhalb(y, gruppe):
    return y - pd.Series(y).groupby(np.asarray(gruppe)).transform("mean").to_numpy()


# ================================================================ 1. Probentypen
print("1 Probentypen")
GRUPPE_PT = {"01": "Primärtumor", "03": "Primärtumor", "05": "Primärtumor",
             "11": "Normalgewebe", "06": "Metastase", "07": "Metastase", "02": "Rezidiv"}
FARBE_PT = {"Primärtumor": BLAU, "Normalgewebe": AQUA, "Metastase": ORANGE, "Rezidiv": ROSA}
pt = meta_all["sample_type"].map(GRUPPE_PT)
tab = pd.crosstab(meta_all["study"], pt)[list(FARBE_PT)]
tab = tab.loc[tab.sum(axis=1).sort_values(ascending=False).index]
fig, ax = plt.subplots(figsize=(5.8, 2.45))
unten = np.zeros(len(tab))
for g, farbe in FARBE_PT.items():
    ax.bar(tab.index, tab[g], bottom=unten, color=farbe, width=0.74, label=f"{g} ({de(tab[g].sum())})",
           edgecolor="white", linewidth=0.5)
    unten += tab[g].to_numpy()
ax.set_ylabel("Proben")
ax.set_xlim(-0.7, len(tab) - 0.3)
ax.tick_params(axis="x", labelrotation=90, length=0, labelsize=6.8)
ax.yaxis.set_major_formatter(lambda v, _: de(v))
ax.legend(ncol=4, loc="upper right", fontsize=7, handlelength=1, handleheight=0.8, columnspacing=1.2)
speichern(fig, "probentypen.pdf")

# ================================================================ 2. Sequenziergerät: welche Tumortypen haben Proben vom älteren Gerät?
print("2 Sequenziergerät")
NAMEN_GA = {"UCEC": "Gebärmutterkörper (UCEC)", "COAD": "Dickdarm (COAD)", "READ": "Mastdarm (READ)"}
plattform = meta["platform"].to_numpy()
fig, ax = plt.subplots(figsize=(5.0, 1.45))
y = np.arange(len(NAMEN_GA))[::-1]
for yi, t in zip(y, NAMEN_GA):
    m = study == t
    n_hi, n_ga = int((m & (plattform == "HiSeq")).sum()), int((m & (plattform == "GA")).sum())
    ax.barh(yi, n_hi, color=GRAU_SERIE, height=0.62, label="Illumina HiSeq" if yi == y[0] else None)
    ax.barh(yi, n_ga, left=n_hi, color=ORANGE, height=0.62, label="Genome Analyzer (älter)" if yi == y[0] else None)
    ax.text(n_hi + n_ga + 8, yi, f"{de(n_ga)} von {de(n_hi + n_ga)} vom älteren Gerät", va="center", fontsize=6.6, color=TEXT)
    print(f"  {t}: {n_ga} von {n_hi + n_ga} vom älteren Gerät")
ax.set_yticks(y, list(NAMEN_GA.values()))
ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 800)
ax.set_xticks([])
ax.grid(False)
ax.spines["bottom"].set_visible(False)
ax.legend(loc="lower right", fontsize=6.6, handlelength=1)
speichern(fig, "geraet.pdf")
print(f"  Proben vom älteren Gerät insgesamt: {(plattform == 'GA').sum()}")

# ================================================================ 3. Fehlende Werte
print("3 Fehlende Werte")
X_sel = X_all[zeile]                                     # Proben des Standardlaufs, alle Gene, Rohwerte
nan = np.isnan(X_sel)
nan_gen = nan.any(axis=0)
zeilen = []
for t in pd.unique(study):
    m = study == t
    alle = nan[m].all(axis=0)
    teil = nan[m].any(axis=0) & ~alle
    if alle.sum() + teil.sum() > 0:
        zeilen.append((t, alle.sum(), teil.sum(), m.sum()))
fw = pd.DataFrame(zeilen, columns=["typ", "alle", "teil", "n"]).set_index("typ")
fw = fw.loc[(fw["alle"] + fw["teil"]).sort_values().index]
NAMEN_FW = {"LAML": "Leukämie (LAML)", "STAD": "Magen (STAD)", "ESCA": "Speiseröhre (ESCA)", "OV": "Eierstock (OV)",
            "PRAD": "Prostata (PRAD)", "UCEC": "Gebärmutterkörper (UCEC)", "COAD": "Dickdarm (COAD)", "READ": "Mastdarm (READ)"}
fig, ax = plt.subplots(figsize=(5.0, 2.3))
y = np.arange(len(fw))
ax.barh(y, fw["alle"], color=BLAU, height=0.62, label="Wert fehlt bei allen Proben des Typs", edgecolor="white", linewidth=0.5)
ax.barh(y, fw["teil"], left=fw["alle"], color=GELB, height=0.62, label="Wert fehlt bei einem Teil der Proben",
        edgecolor="white", linewidth=0.5)
ax.set_yticks(y, [NAMEN_FW[t] for t in fw.index])
ax.tick_params(axis="y", length=0)
ax.grid(axis="x")
ax.grid(axis="y", visible=False)
ax.xaxis.set_major_formatter(lambda v, _: de(v))
ax.set_xlabel("Gene mit fehlenden Werten")
for i, (t, r) in enumerate(fw.iterrows()):
    ax.text(r["alle"] + r["teil"] + 40, i, de(r["alle"] + r["teil"]), va="center", fontsize=6.8, color=TEXT)
ax.set_xlim(0, fw[["alle", "teil"]].sum(axis=1).max() * 1.12)
ax.legend(loc="lower right", fontsize=6.8, handlelength=1)
speichern(fig, "fehlende_werte.pdf")
print(f"  Gene mit fehlenden Werten: {nan_gen.sum()}")

# ================================================================ 4. Werte vor und nach der Log-Transformation
print("4 Log-Transformation")
R = np.clip(X_sel[:, ~nan_gen], 0, None).astype(np.float64)   # ohne Gene mit fehlenden Werten, negative Werte auf 0
del X_sel, nan
rng = np.random.default_rng(SEED)
stich = rng.choice(len(R), 400, replace=False)


def verteilung(werte, mittel, varianz, log, name):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.8, 2.05), gridspec_kw={"width_ratios": [1.15, 1]})
    if not log:
        a1.hist(werte[werte <= 5000], bins=100, color=BLAU)
        a1.set_xlabel("Wert (abgeschnitten bei 5.000)")
        a1.text(0.97, 0.92, f"{prozent((werte > 5000).mean())} der Werte\nliegen über 5.000",
                transform=a1.transAxes, ha="right", va="top", fontsize=6.5, color=MUTED)
        a1.xaxis.set_major_formatter(lambda v, _: de(v))
    else:
        a1.hist(werte, bins=100, color=BLAU)
        a1.set_xlabel(r"$\log_2(\mathrm{Wert} + 1)$")
    a1.set_title("Verteilung aller Einträge", loc="left")
    a1.set_yticks([])
    a1.grid(False)
    a1.spines["left"].set_visible(False)
    # rechts: Varianz pro Gen gegen Mittelwert pro Gen
    o = np.argsort(varianz)[::-1]
    top = np.zeros(len(varianz), bool)
    top[o[:50]] = True
    a2.scatter(mittel[~top], varianz[~top], s=1.5, color=GRAU_SERIE, alpha=0.5, linewidths=0, rasterized=True)
    a2.scatter(mittel[top], varianz[top], s=5, color=ORANGE, linewidths=0, rasterized=True)
    anteil = varianz[top].sum() / varianz.sum()
    a2.set_title(f"ein Punkt pro Gen; orange: 50 Gene = {prozent(anteil)} der Varianz", loc="left")
    a2.set_xlabel("Mittelwert")
    a2.set_ylabel("Varianz")
    a2.grid(axis="both")
    if not log:
        a2.xaxis.set_major_locator(plt.MaxNLocator(3))
        a2.xaxis.set_major_formatter(lambda v, _: de(v))
        a2.yaxis.set_major_formatter(lambda v, _: de(v / 1e9))
        a2.set_ylabel("Varianz (Milliarden)")
    fig.tight_layout(w_pad=2.2)
    speichern(fig, name)
    print(f"  {name}: Top-50-Anteil an der Gesamtvarianz {anteil:.1%}")


verteilung(R[stich].ravel(), R.mean(axis=0), R.var(axis=0), False, "werte_roh.pdf")

# nur die Verteilung der Einträge (Folie „Die Werte vor der Transformation“)
werte = R[stich].ravel()
fig, ax = plt.subplots(figsize=(5.0, 2.0))
ax.hist(werte[werte <= 5000], bins=100, color=BLAU)
ax.set_xlabel("Wert (abgeschnitten bei 5.000)")
ax.text(0.97, 0.92, f"{prozent((werte > 5000).mean())} der Werte liegen über 5.000,\neinzelne bei mehreren Millionen",
        transform=ax.transAxes, ha="right", va="top", fontsize=6.8, color=MUTED)
ax.xaxis.set_major_formatter(lambda v_, _: de(v_))
ax.set_ylabel("Anzahl Einträge")
ax.set_yticks([])
ax.grid(False)
speichern(fig, "werte_roh_verteilung.pdf")

# nur die Varianz pro Gen gegen den Mittelwert pro Gen (Folie „Die Varianz der Gene vor der Transformation“)
mittel_r, var_r0 = R.mean(axis=0), R.var(axis=0)
top = np.zeros(len(var_r0), bool)
top[np.argsort(var_r0)[::-1][:50]] = True
fig, ax = plt.subplots(figsize=(5.0, 2.3))
ax.scatter(mittel_r[~top], var_r0[~top] / 1e9, s=2, color=GRAU_SERIE, alpha=0.5, linewidths=0, rasterized=True)
ax.scatter(mittel_r[top], var_r0[top] / 1e9, s=6, color=ORANGE, linewidths=0, rasterized=True)
i_alb = list(symbol[~nan_gen]).index("ALB")
ax.annotate("Albumin (ALB)", (mittel_r[i_alb], var_r0[i_alb] / 1e9), xytext=(-8, -2), textcoords="offset points",
            fontsize=6.6, color=TEXT, ha="right", va="center")
ax.text(0.03, 0.92, f"ein Punkt pro Gen ({de(len(var_r0))} Gene)\norange: die 50 Gene mit der größten Varianz",
        transform=ax.transAxes, ha="left", va="top", fontsize=6.8, color=MUTED)
ax.set_xlabel("Mittelwert des Gens")
ax.set_ylabel("Varianz (Milliarden)")
ax.xaxis.set_major_formatter(lambda v_, _: de(v_))
ax.yaxis.set_major_formatter(lambda v_, _: de(v_))
ax.grid(axis="both")
speichern(fig, "varianz_roh.pdf")

# Beispiel für die Varianz: Albumin (ALB) und der Anteil der stärksten Gene an der Gesamtvarianz
sym_r = symbol[~nan_gen]
var_r = R.var(axis=0)
o = np.argsort(var_r)[::-1]
gesamt = var_r.sum()
alb = R[:, list(sym_r).index("ALB")]
leber = np.isin(study, ["LIHC", "CHOL"])
fig, a2 = plt.subplots(figsize=(5.4, 0.95))
teile_v = [("ALB", var_r[o[0]]), ("Gene 2 bis 10", var_r[o[1:10]].sum()), ("Gene 11 bis 50", var_r[o[10:50]].sum()),
           (f"übrige {de(len(o) - 50)} Gene", var_r[o[50:]].sum())]
links = 0
for (name, w_), farbe in zip(teile_v, [ORANGE, BLAU, "#7fb0e8", HELLGRAU]):
    a2.barh(0, w_ / gesamt * 100, left=links, color=farbe, height=0.6, edgecolor="white", linewidth=1)
    a2.text(links + w_ / gesamt * 50, 0, prozent(w_ / gesamt), ha="center", va="center", fontsize=6.8,
            color="white" if farbe != HELLGRAU else TEXT, fontweight="semibold")
    a2.text(links + w_ / gesamt * 50, -0.48, name, ha="center", va="top", fontsize=6.4, color=TEXT)
    links += w_ / gesamt * 100
a2.set_xlim(0, 100)
a2.set_ylim(-0.6, 0.45)
a2.set_title(f"Anteil an der Varianz aller {de(len(o))} Gene", loc="left")
a2.axis("off")
speichern(fig, "varianz_anteil.pdf")
print(f"  ALB: Anteil {var_r[o[0]] / gesamt:.1%}, Leber-Median {np.median(alb[leber]):.0f}, Max {alb.max():.0f}, andere Median {np.median(alb[~leber]):.1f}")
L = np.log2(R + 1)
del R
verteilung(L[stich].ravel(), L.mean(axis=0), L.var(axis=0), True, "werte_log.pdf")

# ================================================================ 5. Varianz der Gene (nach Expressionsfilter und ohne Geschlechtsgene)
print("5 Genauswahl")
sym_ok = symbol[~nan_gen]
anteil = pd.DataFrame(L > 0).groupby(study).mean().max(axis=0).to_numpy()
var = L.var(axis=0)
mittel_zwei = np.array([L[:, list(sym_ok).index(g_)].mean() for g_ in ["ALB", "SERPINC1"]])   # für das Beispiel zur Hauptkomponentenanalyse
filt = (anteil >= 0.5) & (var > 0)
geschlecht = np.isin(sym_ok, GESCHLECHTSGENE)
v = np.sort(var[filt & ~geschlecht])[::-1]
fig, ax = plt.subplots(figsize=(2.75, 2.1))
x = np.arange(1, len(v) + 1)
k90 = int(np.searchsorted(np.cumsum(v) / v.sum(), 0.9)) + 1                 # so viele Gene tragen 90 % der Varianz (wie D2)
ax.fill_between(x[:k90], v[:k90], color=BLAU, alpha=0.25, linewidth=0)
ax.plot(x, v, color=BLAU, lw=1.6)
ax.axvline(k90, color=MUTED, lw=0.8, ls=(0, (3, 2)))
ax.text(k90 + 150, v[0] * 0.8, f"{de(k90)} Gene:\n{prozent(v[:k90].sum() / v.sum())} der\nGesamtvarianz",
        fontsize=7.2, color=TEXT, va="top")
ax.set_xlabel("Gene, nach Varianz sortiert")
ax.set_ylabel(r"Varianz von $\log_2(x+1)$")
ax.xaxis.set_major_formatter(lambda v_, _: de(v_))
ax.xaxis.set_major_locator(plt.MultipleLocator(4000))
ax.set_xlim(0, len(v))
ax.set_ylim(0, None)
speichern(fig, "varianz_gene.pdf")
print(f"  {len(v)} Gene, {k90} Gene tragen {v[:k90].sum() / v.sum():.1%} der Varianz")

# ================================================================ 6. Geschlechtsgene
print("6 Geschlechtsgene")
# r² pro Gen innerhalb der Tumortypen (nur Tumortypen mit beiden Geschlechtern), wie notebook_aufbereitung.ipynb, Abschnitt 4
gender = meta["gender"].to_numpy()
beide = meta.groupby("study")["gender"].nunique()
m = (meta["study"].isin(beide[beide == 2].index) & meta["gender"].notna()).to_numpy()
Lf = L[:, filt]
Rr = Lf[m] - pd.DataFrame(Lf[m]).groupby(study[m]).transform("mean").to_numpy()
w = (gender[m] == "weiblich").astype(float)
w -= w.mean()
r2 = ((w @ Rr) / np.sqrt((w**2).sum() * (Rr**2).sum(axis=0))) ** 2
r2 = pd.Series(r2, index=sym_ok[filt]).sort_values(ascending=False)
del Rr, Lf, L

fig, ax = plt.subplots(figsize=(5.0, 1.95))
top = r2.head(20)
farben = np.where(top.to_numpy() > 0.4, BLAU, GRAU_SERIE)
ax.bar(range(len(top)), top.to_numpy(), color=farben, width=0.75)
ax.axhline(0.4, color=MUTED, lw=0.8, ls=(0, (3, 2)))
ax.text(len(top) - 0.5, 0.42, "Schwelle 0,4", ha="right", va="bottom", fontsize=6.5, color=MUTED)
ax.set_xticks(range(len(top)), top.index, rotation=90, fontsize=6.2)
ax.tick_params(axis="x", length=0)
ax.set_ylabel("durch das Geschlecht\nerklärter Anteil $r^2$")
ax.yaxis.set_major_formatter(lambda v_, _: de(v_, 1))
speichern(fig, "geschlecht.pdf")
eta_nachher = max(eta2(innerhalb(Z[:, j], study), gender) for j in range(20))
print(f"  über Schwelle: {list(top.index[top > 0.4])}; eta² Geschlecht nachher max {eta_nachher:.2%}")
print(f"  Gene des Y-Chromosoms im Datensatz nach B1: {[g for g in ['RPS4Y1', 'DDX3Y', 'KDM5D', 'PRKY', 'NLGN4Y'] if g in set(sym_ok)]}")

# ================================================================ 7. Störgrößen: Tumortyp gegen Platte (notebook_aufbereitung.ipynb, Abschnitt 7)
print("7 Störgrößen")
rng = np.random.default_rng(SEED)
platte = meta["plate"].to_numpy()
platte_zufall = meta.groupby("study")["plate"].transform(lambda s: rng.permutation(s.to_numpy())).to_numpy()
x = np.arange(1, 21)
e_typ = [eta2(Z[:, j], study) for j in range(20)]
e_pl = [eta2(innerhalb(Z[:, j], study), platte) for j in range(20)]
e_zu = [eta2(innerhalb(Z[:, j], study), platte_zufall) for j in range(20)]
e_ge = [eta2(innerhalb(Z[:, j], study), gender) for j in range(20)]
fig, ax = plt.subplots(figsize=(5.6, 2.1))
ax.plot(x, e_typ, color=BLAU, lw=2, marker="o", ms=3.5, label="Tumortyp")
ax.plot(x, e_pl, color=ORANGE, lw=2, marker="o", ms=3.5, label="Platte (innerhalb der Tumortypen)")
ax.plot(x, e_zu, color=ORANGE, lw=1.2, ls=(0, (3, 2)), label="Platte, zufällig vertauscht")
ax.plot(x, e_ge, color=GRAU_SERIE, lw=1.5, marker="o", ms=2.5, label="Geschlecht (innerhalb der Tumortypen)")
ax.set_xticks(x)
ax.set_xlabel("Hauptkomponente")
ax.set_ylabel(r"erklärter Anteil $\eta^2$")
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(lambda v_, _: prozent(v_))
ax.legend(fontsize=6.6, loc="center left", bbox_to_anchor=(1.0, 0.5))
speichern(fig, "stoergroessen.pdf")

# Plattenachse innerhalb des Brustkrebses
b = study == "BRCA"
Zb, Vb, evb = pca(Xc[b] - Xc[b].mean(axis=0), k=10)
pl_b = platte[b]
j_pl = int(np.argmax([eta2(Zb[:, j], pl_b) for j in range(10)]))
mittel = pd.Series(Zb[:, j_pl]).groupby(pl_b).agg(["mean", "size"])
gross = mittel[mittel["size"] >= 15].sort_values("mean")
fig, ax = plt.subplots(figsize=(5.8, 1.95))
daten = [Zb[pl_b == q, j_pl] for q in gross.index]
bp = ax.boxplot(daten, showfliers=False, widths=0.6, patch_artist=True,
                medianprops=dict(color=TEXT, lw=1), whiskerprops=dict(color=MUTED, lw=0.8), capprops=dict(color=MUTED, lw=0.8),
                boxprops=dict(facecolor="#d6e6f7", edgecolor=BLAU, lw=0.8))
for i, d in enumerate(daten):
    ax.scatter(np.full(len(d), i + 1) + rng.uniform(-0.15, 0.15, len(d)), d, s=2, alpha=0.5, color=BLAU, linewidths=0)
ax.set_xticks(range(1, len(gross) + 1), gross.index, rotation=90, fontsize=6)
ax.tick_params(axis="x", length=0)
ax.set_xlabel("Platte (mindestens 15 Proben)")
ax.set_ylabel(f"Hauptkomponente {j_pl + 1}")
speichern(fig, "platte_brustkrebs.pdf")
print(f"  Brustkrebs: HK {j_pl + 1} ({evb[j_pl]:.1%}), Platte erklärt {eta2(Zb[:, j_pl], pl_b):.0%}")

# ================================================================ PCA
print("9 Hauptkomponentenanalyse")
cum = np.cumsum(ev)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.8, 2.0))
a1.bar(np.arange(1, 51), ev[:50] * 100, color=BLAU, width=0.75)
a1.set_title("Anteil pro Hauptkomponente", loc="left")
a1.set_xlabel("Hauptkomponente")
a1.yaxis.set_major_formatter(lambda v_, _: de(v_) + " %")
a2.plot(np.arange(1, 101), cum * 100, color=BLAU, lw=1.8)
for k in [2, 10, 50]:
    a2.plot(k, cum[k - 1] * 100, "o", color=BLAU, ms=4)
    a2.annotate(f"{k}: {prozent(cum[k - 1])}", (k, cum[k - 1] * 100), xytext=(6, -10), textcoords="offset points",
                fontsize=6.6, color=TEXT)
a2.set_title("kumuliert", loc="left")
a2.set_xlabel("Zahl der Hauptkomponenten")
a2.set_ylim(0, 100)
a2.yaxis.set_major_formatter(lambda v_, _: de(v_) + " %")
fig.tight_layout(w_pad=2)
speichern(fig, "pca_varianz.pdf")

# Idee der Hauptkomponentenanalyse: Projektion auf eine Richtung (Länge 1), Varianz der Projektion vergleichen
var_gen = Xc.astype(np.float64).var(axis=0, ddof=1)
T_gesamt = var_gen.sum()
r_zuf = np.random.default_rng(SEED).normal(size=Xc.shape[1])
r_zuf /= np.linalg.norm(r_zuf)
j_max = int(np.argmax(var_gen))
RICHTUNGEN = [("eine zufällige Richtung", Xc.astype(np.float64) @ r_zuf),
              (f"das Gen mit der größten Varianz ({genes['name'].iloc[j_max]})", Xc[:, j_max].astype(np.float64))]
fig, axes = plt.subplots(3, 1, figsize=(5.9, 1.85), sharex=True)
rng_i = np.random.default_rng(SEED)
farben_i = np.array([FARBE[t] for t in study])
for ax, (titel, proj) in zip(axes, RICHTUNGEN):
    ax.scatter(proj, rng_i.uniform(-1, 1, len(proj)), s=1.2, c=farben_i, alpha=0.6, linewidths=0, rasterized=True)
    v_ = proj.var(ddof=1)
    ax.set_title(titel, loc="left", fontsize=7, pad=2)
    ax.text(1.0, 0.5, f"Varianz {de(v_, 1)}\n= {de(100 * v_ / T_gesamt, 2)} % von T",
            transform=ax.transAxes, ha="left", va="center", fontsize=6.8, color=TEXT)
for ax in axes:
    ax.set_yticks([]); ax.set_ylim(-1.6, 1.6)
    ax.spines["left"].set_visible(False)
    ax.grid(False)
axes[2].set_title("gesucht: eine gewichtete Kombination aller Gene mit möglichst großer Varianz", loc="left", fontsize=7, pad=2,
                  color=BLAU)
axes[2].text(0.5, 0.5, "?", transform=axes[2].transAxes, ha="center", va="center", fontsize=14, color=BLAU, fontweight="semibold")
axes[-1].set_xlabel("Koordinate auf der Richtung (Score)")
axes[-1].tick_params(labelsize=6.5)
fig.tight_layout(h_pad=0.4)
speichern(fig, "pca_idee.pdf")
print(f"  totale Varianz T = {T_gesamt:.0f}; zufällig {RICHTUNGEN[0][1].var(ddof=1):.2f}, bestes Gen {var_gen[j_max]:.1f}")

# Hauptkomponente 1 allein: Verteilung pro Tumortyp, sortiert nach dem Median
pc1 = pd.Series(Z[:, 0]).groupby(study)
reihe = pc1.median().sort_values().index.tolist()
fig, ax = plt.subplots(figsize=(5.9, 2.1))
bp = ax.boxplot([Z[study == t, 0] for t in reihe], widths=0.65, patch_artist=True, showfliers=False,
                medianprops=dict(color=TEXT, lw=1), whiskerprops=dict(color=MUTED, lw=0.6), capprops=dict(color=MUTED, lw=0.6))
for box, t in zip(bp["boxes"], reihe):
    box.set(facecolor=FARBE[t], edgecolor=FARBE[t], alpha=0.85)
ax.axhline(0, color=HELLGRAU, lw=0.6, zorder=0)
ax.set_xticks(np.arange(1, len(reihe) + 1), reihe, rotation=90, fontsize=6.3)
ax.tick_params(axis="x", length=0)
ax.set_ylabel(f"Hauptkomponente 1 ({prozent(ev[0])})", fontsize=7)
ax.tick_params(axis="y", labelsize=6)
ax.grid(axis="x", visible=False)
legende_gruppen(ax, loc="upper left", fontsize=6, ncol=2)
speichern(fig, "pca_1.pdf")
print("  Median Hauptkomponente 1:", pc1.median().sort_values().round(0).to_dict())

# Hauptkomponenten 1+2 und 3+4: farbig nur die Tumortypen, die diese zwei Achsen abspalten
# Kriterium: mindestens 45 % der 15 nächsten Nachbarn (in den zwei Achsen) aus dem eigenen Typ
NAME_LANG = {
    "GBM": "Glioblastom", "LGG": "Niedriggradiges Gliom", "PCPG": "Phäochromozytom/Paragangliom",
    "ACC": "Nebennierenrindenkarzinom", "UVM": "Uveamelanom (Auge)", "SKCM": "Melanom der Haut",
    "LAML": "Akute myeloische Leukämie", "DLBC": "Diffus großzelliges B-Zell-Lymphom", "THYM": "Thymom",
    "TGCT": "Keimzelltumor des Hodens", "KIRC": "Klarzelliges Nierenzellkarzinom", "KIRP": "Papilläres Nierenzellkarzinom",
    "KICH": "Chromophobes Nierenzellkarzinom", "LIHC": "Leberzellkarzinom", "CHOL": "Gallengangskarzinom",
    "PAAD": "Pankreaskarzinom", "STAD": "Magenkarzinom", "ESCA": "Speiseröhrenkarzinom", "COAD": "Dickdarmkarzinom",
    "READ": "Mastdarmkarzinom", "HNSC": "Plattenepithelkarzinom Kopf-Hals", "LUSC": "Plattenepithelkarzinom Lunge",
    "CESC": "Gebärmutterhalskarzinom", "BLCA": "Harnblasenkarzinom", "LUAD": "Adenokarzinom Lunge", "MESO": "Mesotheliom",
    "THCA": "Schilddrüsenkarzinom", "PRAD": "Prostatakarzinom", "BRCA": "Brustkrebs", "OV": "Eierstockkarzinom",
    "UCEC": "Endometriumkarzinom", "UCS": "Karzinosarkom der Gebärmutter", "SARC": "Sarkom (Weichgewebe)",
}
SCHWELLE_ABGESPALTEN = 0.45
FARBE_ABGESPALTEN = {"LGG": BLAU, "PCPG": ORANGE, "LIHC": AQUA, "UVM": GELB, "GBM": ROSA, "LAML": "#7d5ba6",   # eine Farbe pro Typ,
                     "KIRC": "#c0392b", "KIRP": "#f08a7e", "PRAD": "#34495e", "DLBC": "#16a085"}                  # gleich auf beiden Folien
EPITHEL_FARBEN = {"COAD": "#8c2d04", "READ": "#d9480f", "STAD": "#a6761d", "ESCA": "#c9a227",   # Drüsen: warme Töne
                  "HNSC": "#1b4f72", "LUSC": "#4f8fc0", "CESC": "#17a2b8", "BLCA": "#5c6b2e"}   # Plattenepithel, Harnblase
for (i, j), name in [((0, 1), "pca_12.pdf"), ((2, 3), "pca_34.pdf")]:
    Y2 = Z[:, [i, j]]
    nb2 = nachbarn(Y2)
    rein2 = pd.Series((study[nb2] == study[:, None]).mean(axis=1)).groupby(study).mean().sort_values(ascending=False)
    farbig = rein2[rein2 >= SCHWELLE_ABGESPALTEN].index.tolist()
    print(f"  {name}: abgespalten {rein2[farbig].round(2).to_dict()}, nächste darunter {rein2.drop(farbig).head(2).round(2).to_dict()}")
    fig, ax = plt.subplots(figsize=(4.4, 3.5))
    grau = ~np.isin(study, farbig)
    if name == "pca_12.pdf":                                         # Epithelkarzinome: alle Tumoren dieser Typen
        epi = np.isin(study, list(EPITHEL_FARBEN)) & grau
        grau = grau & ~epi
    ax.scatter(Y2[grau, 0], Y2[grau, 1], s=2.2, color=GRAU_SERIE, alpha=0.35, linewidths=0, rasterized=True,
               label="übrige Tumoren, noch überlappend")
    if name == "pca_12.pdf":
        # jedes Epithelkarzinom in eigener Farbe
        for t, f in EPITHEL_FARBEN.items():
            m = epi & (study == t)
            ax.scatter(Y2[m, 0], Y2[m, 1], s=2.6, color=f, alpha=0.6, linewidths=0, rasterized=True)
        epi_handles = [plt.Line2D([], [], marker="o", ls="", ms=4, color=f, label=f"{NAME_LANG[t]} ({t})") for t, f in EPITHEL_FARBEN.items()]
        # Ellipse um 90 % der Epithelkarzinome
        P = Y2[epi]
        mitte_e = P.mean(axis=0)
        ew, evec = np.linalg.eigh(np.cov(P.T))
        r_e = np.sqrt(np.quantile(np.einsum("ij,jk,ik->i", P - mitte_e, np.linalg.inv(np.cov(P.T)), P - mitte_e), 0.90))
        winkel = np.degrees(np.arctan2(evec[1, 1], evec[0, 1]))
        breite_e, hoehe_e = 2 * r_e * np.sqrt(ew[1]), 2 * r_e * np.sqrt(ew[0])
        ax.add_patch(mpl.patches.Ellipse(mitte_e, breite_e, hoehe_e, angle=winkel, fill=False, ec=TEXT, lw=1.0, ls="--", zorder=5))
        t_e = np.linspace(0, 2 * np.pi, 400)
        rand = mitte_e + (np.c_[np.cos(t_e) * breite_e / 2, np.sin(t_e) * hoehe_e / 2]
                          @ np.array([[np.cos(np.radians(winkel)), np.sin(np.radians(winkel))],
                                      [-np.sin(np.radians(winkel)), np.cos(np.radians(winkel))]]))
        linke_haelfte = rand[rand[:, 0] < mitte_e[0]]
        links_e = linke_haelfte[np.argmin(np.abs(linke_haelfte[:, 1] + 25))]   # linker Rand auf Höhe −25, zwischen zwei Achsenzahlen
    for t, f in ((t_, FARBE_ABGESPALTEN[t_]) for t_ in farbig):
        m = study == t
        ax.scatter(Y2[m, 0], Y2[m, 1], s=3.0, color=f, alpha=0.7, linewidths=0, rasterized=True, label=f"{NAME_LANG[t]} ({t})")
    ax.set_xlabel(f"Hauptkomponente {i + 1} ({prozent(ev[i])})")
    ax.set_ylabel(f"Hauptkomponente {j + 1} ({prozent(ev[j])})")
    ax.grid(axis="y")
    # abgespaltene Typen direkt an ihrer Punktwolke beschriften (Versatz von Hand, damit sich nichts überdeckt)
    VERSATZ, UEBRIGE = {
        "pca_12.pdf": ({"LGG": (10, 40, "center"), "GBM": (-10, 52, "center"), "PCPG": (0, -52, "center"),
                        "LIHC": (25, -55, "left"), "LAML": (-20, 0, "right"), "UVM": (18, -10, "left")}, (55, 72)),
        "pca_34.pdf": ({"LAML": (0, 22, "center"), "DLBC": (-12, 22, "right"), "KIRC": (10, 32, "left"),
                        "KIRP": (-12, -28, "center"), "PRAD": (18, -28, "left")}, (55, 20)),
    }[name]
    for t, f in ((t_, FARBE_ABGESPALTEN[t_]) for t_ in farbig):
        mx_, my_ = np.median(Y2[study == t], axis=0)
        dx, dy, ha_ = VERSATZ.get(t, (0, 30, "center"))
        ax.text(mx_ + dx, my_ + dy, "Akute myeloische\nLeukämie" if t == "LAML" else NAME_LANG[t], color=f, fontsize=7,
                fontweight="semibold", ha=ha_, va="center", zorder=7,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.text(*UEBRIGE, "übrige Tumoren", color=MUTED, fontsize=7, ha="center", va="center", zorder=7,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.xaxis.label.set_fontsize(8.5); ax.yaxis.label.set_fontsize(8.5)
    if name == "pca_12.pdf":
        # Epithelkarzinome: eigene Legende links, Pfeil zur Ellipse
        y_pf = ax.transAxes.inverted().transform(ax.transData.transform(links_e))[1]
        leg2 = ax.legend(handles=epi_handles, title="Magen-Darm-, Plattenepithel-\nund Harnblasenkarzinome", loc="center right", bbox_to_anchor=(-0.2, y_pf),
                         fontsize=7.2, title_fontsize=8, handletextpad=0.2, borderaxespad=0)
        leg2.get_title().set_fontweight("semibold")
        leg2._legend_box.align = "left"
        ax.annotate("", xy=links_e, xycoords="data", xytext=(-0.19, y_pf), textcoords="axes fraction", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-|>", color=TEXT, lw=1.0, shrinkA=0, shrinkB=1), zorder=6)
        ax.yaxis.set_label_coords(-0.1, 0.28)
    ax.tick_params(labelsize=7.5)
    speichern(fig, name)

# Ladungen: Gene mit den größten Gewichten an beiden Enden
PROGRAMME = {0: ("Schleimhaut, Plattenepithel", "Nervengewebe"), 1: ("Leber, Niere", "Plattenepithel, Bindegewebe"),  # (Ende −, Ende +)
             2: ("Niere, Leber", "Plattenepithel, Nervengewebe"), 3: ("Drüsenepithel", "Immunzellen")}
def ladungen(komponenten, datei):
    fig, axes = plt.subplots(1, len(komponenten), figsize=(5.9, 2.35), squeeze=False)
    for j, ax in zip(komponenten, axes[0]):
        o = np.argsort(Vt[j])
        k = np.r_[o[:5], o[-5:]]
        werte = Vt[j, k]
        ax.barh(np.arange(10), werte, color=np.where(werte < 0, ORANGE, BLAU), height=0.7)
        ax.set_yticks(np.arange(10), genes["name"].to_numpy()[k], fontsize=6)
        ax.tick_params(axis="y", length=0)
        ax.axvline(0, color=MUTED, lw=0.6)
        ax.set_xticks([])
        ax.grid(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_title(f"Hauptkomponente {j + 1}", loc="left", pad=18)
        neg, pos = PROGRAMME[j]
        ax.text(0.0, 1.02, "+ " + pos, transform=ax.transAxes, ha="left", va="bottom", fontsize=6, color=BLAU, fontweight="semibold")
        ax.text(0.0, -0.03, "− " + neg, transform=ax.transAxes, ha="left", va="top", fontsize=6, color=ORANGE, fontweight="semibold")
        lim = np.abs(werte).max() * 1.05
        ax.set_xlim(-lim, lim)
    fig.tight_layout(w_pad=0.6)
    speichern(fig, datei)


ladungen([0, 1, 2, 3], "pca_ladungen.pdf")

# Beispiel Dimensionsreduktion: zwei Lebergene, die sich gemeinsam ändern, lassen sich durch eine Achse beschreiben
namen = genes["name"].tolist()
G2 = Xc[:, [namen.index("ALB"), namen.index("SERPINC1")]].astype(np.float64)       # zentriert, log2
_, s2, v2 = np.linalg.svd(G2, full_matrices=False)
anteil1 = s2[0] ** 2 / (s2**2).sum()
richtung = v2[0] * np.sign(v2[0, 0])
leber = np.isin(study, ["LIHC", "CHOL"])
fig, ax = plt.subplots(figsize=(3.3, 2.6))
ax.scatter(G2[~leber, 0], G2[~leber, 1], s=1.5, color=GRAU_SERIE, alpha=0.4, linewidths=0, rasterized=True, label="andere Tumortypen")
ax.scatter(G2[leber, 0], G2[leber, 1], s=2.5, color=ORANGE, alpha=0.7, linewidths=0, rasterized=True, label="Leber und Gallengang")
ax.text(0.04, 0.96, f"Korrelation {de(np.corrcoef(G2.T)[0, 1], 2)}:\nbeide steigen und fallen gemeinsam", transform=ax.transAxes,
        fontsize=6.6, color=BLAU, ha="left", va="top", fontweight="semibold")
ax.set_xlabel("Albumin (ALB), log$_2$, zentriert")
ax.set_ylabel("Antithrombin (SERPINC1), log$_2$, zentriert")
ax.set_aspect("equal", adjustable="datalim")
ax.legend(loc="lower right", fontsize=6.2, markerscale=3, handletextpad=0.2)
speichern(fig, "zwei_gene.pdf")
print(f"  ALB/SERPINC1: Korrelation {np.corrcoef(G2.T)[0, 1]:.2f}, erste Achse {anteil1:.1%}")

# Hauptkomponentenanalyse Schritt für Schritt an denselben zwei Genen
roh2 = G2 + mittel_zwei                                                            # log2-Werte vor dem Zentrieren
score = G2 @ richtung
fig, axes = plt.subplots(1, 3, figsize=(5.9, 2.05), gridspec_kw={"width_ratios": [1, 1, 1.15]})
for ax, D_, titel in [(axes[0], roh2, "1. Zentrieren"), (axes[1], G2, "2. Achsen finden")]:
    ax.scatter(D_[~leber, 0], D_[~leber, 1], s=0.8, color=GRAU_SERIE, alpha=0.35, linewidths=0, rasterized=True)
    ax.scatter(D_[leber, 0], D_[leber, 1], s=1.2, color=ORANGE, alpha=0.6, linewidths=0, rasterized=True)
    ax.set_title(titel, loc="left")
    ax.set_aspect("equal", adjustable="datalim")
    ax.tick_params(labelsize=5.5)
    ax.set_xlabel("Albumin (ALB)", fontsize=6.2)
axes[0].set_ylabel("Antithrombin (SERPINC1)", fontsize=6.2)
m0 = mittel_zwei
axes[0].plot(*m0, marker="+", ms=9, mew=1.6, color=BLAU)
axes[0].annotate("Mittelwert", m0, xytext=(m0[0] + 4, m0[1] - 5), fontsize=5.8, color=BLAU,
                 arrowprops=dict(arrowstyle="-", color=BLAU, lw=0.6))
axes[1].axhline(0, color=HELLGRAU, lw=0.5)
axes[1].axvline(0, color=HELLGRAU, lw=0.5)
zweite = np.array([-richtung[1], richtung[0]])
axes[1].annotate("", xy=richtung * 16, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=BLAU, lw=1.6))
axes[1].annotate("", xy=zweite * 6, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=TEXT, lw=1.2))
axes[1].text(*(richtung * 16 + np.array([-0.5, 1.5])), f"Achse 1\n{prozent(anteil1)}", fontsize=5.8, color=BLAU, ha="right", va="bottom", fontweight="semibold")
axes[1].text(*(zweite * 6 + np.array([-0.5, 0.5])), f"Achse 2\n{prozent(1 - anteil1)}", fontsize=5.8, color=TEXT, ha="right", va="bottom")
a3 = axes[2]
rng_s = np.random.default_rng(SEED)
a3.scatter(score[~leber], rng_s.uniform(-1, 1, (~leber).sum()), s=0.8, color=GRAU_SERIE, alpha=0.35, linewidths=0, rasterized=True)
a3.scatter(score[leber], rng_s.uniform(-1, 1, leber.sum()), s=1.2, color=ORANGE, alpha=0.6, linewidths=0, rasterized=True)
a3.set_ylim(-3.2, 1.6)
a3.set_yticks([])
a3.spines["left"].set_visible(False)
a3.grid(False)
a3.set_title("3. Score: eine Zahl pro Tumor", loc="left")
a3.set_xlabel("Score auf Achse 1", fontsize=6.2)
a3.tick_params(labelsize=5.5)
gew = f"Score = {de(richtung[0], 2)} · ALB + {de(richtung[1], 2)} · SERPINC1"
a3.text(0.5, 0.06, gew, transform=a3.transAxes, ha="center", va="bottom", fontsize=5.9, color=BLAU)
fig.tight_layout(w_pad=1.0)
speichern(fig, "pca_schritte.pdf")
print(f"  Ladungen Achse 1: ALB {richtung[0]:.2f}, SERPINC1 {richtung[1]:.2f}; Mittelwerte {m0.round(2)}")

# ================================================================ UMAP
print("10 UMAP")
fig, ax = plt.subplots(figsize=(5.8, 3.1))
streubild(ax, E, study, s=3.0, alpha=0.5, label_size=6.8)
ohne_achsen(ax)
legende_gruppen(ax, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=6, fontsize=6.3, columnspacing=0.8)
speichern(fig, "umap.pdf")

# Ziel der Dimensionsreduktion: vorher Tabelle ohne sichtbare Struktur, nachher Bild mit Gruppen
fig = plt.figure(figsize=(5.9, 2.55))
a1 = fig.add_axes([0.0, 0.08, 0.36, 0.8])
a2 = fig.add_axes([0.5, 0.02, 0.5, 0.86])
# echter Ausschnitt der aufbereiteten Matrix: erste Zeilen und Spalten, dazu die letzte Spalte
n_z, m_z = 8, 4
benannt = np.flatnonzero(~genes["name"].astype(str).str.startswith("?").to_numpy())      # Gene ohne Symbol überspringen
spalten_z = list(benannt[:m_z]) + [benannt[-1]]
kopf = [genes["name"].iloc[j] for j in spalten_z]
a1.set_xlim(0, m_z + 2.6); a1.set_ylim(n_z + 2, -1.2); a1.axis("off")
x_sp = [1.6 + j for j in range(m_z)] + [m_z + 2.1]
for j, (x_, k_) in enumerate(zip(x_sp, kopf)):
    a1.text(x_, -0.5, k_, ha="center", va="center", fontsize=6.2, fontweight="semibold", color=TEXT, rotation=0)
a1.text(m_z + 1.1, -0.5, "…", ha="center", va="center", fontsize=6.5, color=MUTED)
a1.plot([0.1, m_z + 2.6], [0.05, 0.05], color=MUTED, lw=0.6)
for i in range(n_z):
    a1.text(0.1, i + 0.6, f"Tumor {i + 1}", ha="left", va="center", fontsize=6.2, color=MUTED)
    for x_, j in zip(x_sp, spalten_z):
        a1.text(x_, i + 0.6, de(Xc[i, j], 1), ha="center", va="center", fontsize=6.2, color=TEXT)
    a1.text(m_z + 1.1, i + 0.6, "…", ha="center", va="center", fontsize=6.5, color=MUTED)
for x_ in [0.55] + x_sp:
    a1.text(x_, n_z + 0.6, "⋮", ha="center", va="center", fontsize=7, color=MUTED, fontfamily="DejaVu Sans")
a1.text(0.1, n_z + 1.6, "9.720 Tumoren × 9.089 Gene, log-transformiert und zentriert", ha="left", va="center", fontsize=5.6, color=MUTED)
a1.set_title("vorher: eine Tabelle mit 9.089 Spalten", loc="left", fontsize=7.5)

streubild(a2, E, study, s=1.2, alpha=0.5, label_size=5.6)
ohne_achsen(a2)
a2.set_title("nachher: 2 Dimensionen, ähnliche Tumoren nah beieinander", loc="left", fontsize=7.5)
fig.add_artist(mpl.patches.FancyArrowPatch((0.37, 0.5), (0.48, 0.5), transform=fig.transFigure, arrowstyle="-|>",
                                           mutation_scale=14, color=BLAU, lw=2))
fig.text(0.425, 0.56, "Dimensions-\nreduktion", ha="center", va="bottom", fontsize=6.5, color=BLAU, fontweight="semibold")
speichern(fig, "ziel_dimred.pdf")

# Redundanz ohne bestimmtes Verfahren: stärkste Korrelation jedes Gens mit einem anderen Gen
S_r = Xc / Xc.std(axis=0)
C_r = (S_r.T @ S_r) / (len(S_r) - 1)
np.fill_diagonal(C_r, 0)
max_r = np.abs(C_r).max(axis=0)
del S_r, C_r
print("  Anteil der Gene mit einem Partner |r| >=", {t: round(float((max_r >= t).mean()), 3) for t in [0.5, 0.6, 0.7, 0.8, 0.9]})

# Beispiel aus den Daten: mittlerer quadrierter Abstand eines Paares = Variation innerhalb beider Typen (Sockel)
# + quadrierter Abstand der Typmittelwerte (Signal)
def innen(t):
    Y_ = Xc[study == t].astype(np.float64)
    return ((Y_ - Y_.mean(axis=0)) ** 2).sum() / (len(Y_) - 1)
def mitte(t):
    return Xc[study == t].astype(np.float64).mean(axis=0)
PAARE_B = [("LIHC", "LIHC", "zwei Lebertumoren"), ("COAD", "READ", "Dickdarm und Mastdarm"),
           ("LUAD", "LUSC", "Lunge: Adeno- und Plattenepithelkarzinom"), ("LIHC", "BRCA", "Leber und Brust")]
zeilen_b = []
for a_, b_, name_b in PAARE_B:
    sockel = innen(a_) + innen(b_) if a_ != b_ else 2 * innen(a_)
    signal = 0.0 if a_ == b_ else float(((mitte(a_) - mitte(b_)) ** 2).sum())
    zeilen_b.append((name_b, sockel, signal))
    print(f"  {name_b}: Sockel {sockel:.0f}, Signal {signal:.0f}, Anteil Signal {signal / (sockel + signal):.1%}, "
          f"Abstand {np.sqrt(sockel + signal):.0f}")
fig, ax = plt.subplots(figsize=(5.6, 1.6))
y_b = np.arange(len(zeilen_b))[::-1]
for y_, (name_b, sockel, signal) in zip(y_b, zeilen_b):
    ax.barh(y_, sockel, color=GRAU_SERIE, height=0.62, label="innerhalb der Typen (Sockel)" if y_ == y_b[0] else None)
    ax.barh(y_, signal, left=sockel, color=BLAU, height=0.62, label="zwischen den Typen (Signal)" if y_ == y_b[-1] else None)
    ax.text(sockel / 2, y_, prozent(sockel / (sockel + signal)), ha="center", va="center", fontsize=6.5, color="white", fontweight="semibold")
    if signal / (sockel + signal) > 0.1:
        ax.text(sockel + signal / 2, y_, prozent(signal / (sockel + signal)), ha="center", va="center", fontsize=6.5, color="white", fontweight="semibold")
ax.set_yticks(y_b, [z[0] for z in zeilen_b], fontsize=7)
ax.tick_params(axis="y", length=0)
ax.xaxis.set_major_formatter(lambda v, _: de(v))
ax.set_xlabel("quadrierter Abstand, Mittel über alle Paare von Tumoren", fontsize=6.5)
ax.tick_params(axis="x", labelsize=6)
ax.grid(axis="y", visible=False)
ax.legend(loc="upper right", fontsize=6.2, handlelength=1)
speichern(fig, "beispiel_rauschen.pdf")

# Abstände im Raum aller Gene: relativ zum mittleren Abstand jedes Tumors, getrennt nach gleichem und anderem Tumortyp
rng_a = np.random.default_rng(SEED)
i_a = rng_a.choice(len(Xc), 500, replace=False)
Y_a = Xc.astype(np.float64)
sq_a = (Y_a**2).sum(axis=1)
d_a = np.sqrt(np.maximum(sq_a[i_a][:, None] - 2 * Y_a[i_a] @ Y_a.T + sq_a[None], 0))
d_a[np.arange(len(i_a)), i_a] = np.nan
r_a = d_a / np.nanmedian(d_a, axis=1)[:, None]
gleich_a = study[None, :] == study[i_a][:, None]
ok_a = ~np.isnan(r_a)
fig, ax = plt.subplots(figsize=(3.4, 2.4))
bins = np.linspace(0, 2, 81)
ax.hist(r_a[ok_a & ~gleich_a], bins=bins, density=True, color=GRAU_SERIE, alpha=0.6, label="anderer Tumortyp")
ax.hist(r_a[ok_a & gleich_a], bins=bins, density=True, color=BLAU, alpha=0.75, label="gleicher Tumortyp")
naechster = np.nanmin(r_a, axis=1).mean()
ax.axvline(naechster, color=ORANGE, lw=1.2)
ax.text(naechster - 0.03, ax.get_ylim()[1] * 0.97, "nächster\nNachbar\n(Mittel)", color=ORANGE, fontsize=6.2, ha="right", va="top")
ax.set_xlim(0, 2)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("Abstand relativ zum mittleren Abstand des Tumors")
ax.legend(loc="upper right", fontsize=6.2, handlelength=1)
speichern(fig, "abstaende.pdf")
print(f"  Abstände (alle Gene): gleicher Typ Median {np.median(r_a[ok_a & gleich_a]):.2f}, anderer Typ {np.median(r_a[ok_a & ~gleich_a]):.2f}, "
      f"nächster Nachbar {naechster:.2f}, 5-95 % {np.nanpercentile(r_a, [5, 95]).round(2)}")
print(f"  Nachbarn aus demselben Typ mit allen Genen: {pd.Series((study[nachbarn(Xc)] == study[:, None]).mean(axis=1)).groupby(study).mean().mean():.3f}")

# Nachbarschaft: Anteil der 15 nächsten Nachbarn (alle Gene, wie die UMAP-Eingabe) pro Paar von Tumortypen
TYPEN = [t for typen in [GRUPPEN[g][1] for g in GRUPPEN] for t in typen]
paare = pd.crosstab(np.repeat(study, nb_ein.shape[1]), study[nb_ein.ravel()], normalize="index").loc[TYPEN, TYPEN]
eigen = pd.Series((study[nb_ein] == study[:, None]).mean(axis=1)).groupby(study).mean().sort_values()
fig, ax = plt.subplots(figsize=(5.8, 1.9))
ax.bar(eigen.index, eigen * 100, color=[FARBE[t] for t in eigen.index], width=0.72)
ax.axhline(eigen.mean() * 100, color=MUTED, lw=0.8, ls=(0, (3, 2)))
ax.text(-0.4, eigen.mean() * 100 + 2, f"Mittel {prozent(eigen.mean())}", fontsize=6.5, color=MUTED, va="bottom")
ax.set_xlim(-0.7, 32.7)
ax.set_ylim(0, 100)
ax.tick_params(axis="x", labelrotation=90, length=0, labelsize=6.5)
ax.yaxis.set_major_formatter(lambda v_, _: de(v_) + " %")
ax.set_ylabel("Nachbarn aus\ndemselben Tumortyp")
legende_gruppen(ax, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=6, fontsize=6.2, columnspacing=0.8)
speichern(fig, "eigener_anteil.pdf")

ARTEN = {"gleiches Organ": (BLAU, [("READ", "COAD"), ("GBM", "LGG"), ("COAD", "READ"), ("UCS", "UCEC"), ("KICH", "KIRC"),
                                   ("LGG", "GBM"), ("KIRP", "KIRC"), ("LUSC", "LUAD")]),
         "gleiches Gewebe, anderes Organ": (ORANGE, [("ESCA", "STAD"), ("ESCA", "HNSC"), ("STAD", "ESCA"), ("ESCA", "LUSC"), ("CESC", "HNSC")]),
         "gemeinsame Entwicklungsherkunft": (AQUA, [("CHOL", "LIHC"), ("UCS", "OV"), ("CESC", "UCEC")])}
fig, ax = plt.subplots(figsize=(4.3, 2.75))
y, beschr = 0, []
for art, (farbe, liste) in ARTEN.items():
    werte = [(a_, b_, paare.loc[a_, b_]) for a_, b_ in liste]
    for i, (a_, b_, w_) in enumerate(werte):
        ax.barh(-y, w_ * 100, color=farbe, height=0.72, label=art if i == 0 else None)
        ax.text(w_ * 100 + 1, -y, prozent(w_), va="center", fontsize=6.3, color=TEXT)
        beschr.append((-y, f"{a_} → {b_}"))
        y += 1
    y += 0.6
ax.set_yticks([p_ for p_, _ in beschr], [t_ for _, t_ in beschr], fontsize=6.6)
ax.tick_params(axis="y", length=0)
ax.set_xticks([])
ax.grid(False)
ax.spines["bottom"].set_visible(False)
ax.set_xlim(0, 80)
ax.legend(loc="lower right", fontsize=6.6, handlelength=1)
speichern(fig, "paare.pdf")
print(f"  Anteil eigener Typ im Mittel {eigen.mean():.1%}")

# Untertypen: Speiseröhre und Gebärmutterhals nach Histologie, Brustkrebs nach ESR1
histologie = meta["patient"].map(cdr["histological_type"]).astype(str).to_numpy()
platten_epi = np.char.find(histologie.astype(str), "Squamous") >= 0
esr1 = Xc[:, genes.index[genes["name"] == "ESR1"][0]]
fig, axes = plt.subplots(1, 3, figsize=(5.9, 2.15))
for ax, t, titel in [(axes[0], "ESCA", "Speiseröhre (ESCA)"), (axes[1], "CESC", "Gebärmutterhals (CESC)")]:
    m = study == t
    ax.scatter(E[~m, 0], E[~m, 1], s=0.6, color=HELLGRAU, linewidths=0, rasterized=True)
    ax.scatter(E[m & platten_epi, 0], E[m & platten_epi, 1], s=3.5, color=BLAU, linewidths=0, label="Plattenepithelkarzinom")
    ax.scatter(E[m & ~platten_epi, 0], E[m & ~platten_epi, 1], s=3.5, color=ORANGE, linewidths=0, label="Adenokarzinom und andere")
    ax.set_title(titel, loc="left")
    ohne_achsen(ax)
axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.02), ncol=2, fontsize=6.5, markerscale=2.2, handletextpad=0.1)
m = study == "BRCA"
axes[2].scatter(E[~m, 0], E[~m, 1], s=0.6, color=HELLGRAU, linewidths=0, rasterized=True)
sc = axes[2].scatter(E[m, 0], E[m, 1], s=3.5, c=esr1[m], cmap="Blues", vmin=-6, vmax=6, linewidths=0)
axes[2].set_title("Brust (BRCA): Östrogenrezeptor", loc="left")
ohne_achsen(axes[2])
cb = fig.colorbar(sc, ax=axes[2], orientation="horizontal", shrink=0.7, pad=0.02, aspect=25)
cb.ax.tick_params(labelsize=5.5)
cb.set_label("ESR1 (log2, zentriert)", fontsize=6)
speichern(fig, "untertypen.pdf")

# Robustheit: n_neighbors × min_dist (wie notebook_umap.ipynb, Abschnitt 5.1) und Zufallsstartwert
CACHE = DATA_DIR / "praesentation_umap_varianten_gene.npz"
varianten = [(k, d) for k in [5, 15, 50] for d in [0.1, 0.5]]
if CACHE.exists():
    cache = dict(np.load(CACHE))
else:
    import umap
    Y = Xc.astype(np.float32)                                    # Eingabe wie im Notebook: alle Gene
    cache = {}
    for k, d in varianten:
        print(f"  UMAP n_neighbors={k}, min_dist={d}")
        cache[f"k{k}_d{d}"] = umap.UMAP(n_components=2, n_neighbors=k, min_dist=d, random_state=SEED).fit_transform(Y)
    for seed in [1, 2]:
        print(f"  UMAP Zufallsstartwert {seed}")
        cache[f"seed{seed}"] = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.5, random_state=seed).fit_transform(Y)
    np.savez(CACHE, **cache)

fig, axes = plt.subplots(2, 3, figsize=(5.8, 3.05))
for (k, d), ax in zip([(k, d) for d in [0.1, 0.5] for k in [5, 15, 50]], axes.ravel()):
    streubild(ax, cache[f"k{k}_d{d}"], study, s=0.5, beschriften=False, alpha=0.6)
    ohne_achsen(ax)
    ax.set_title(f"n_neighbors = {k}, min_dist = {de(d, 1)}", fontsize=6.5, fontweight="normal",
                 color=TEXT if (k, d) != (15, 0.5) else BLAU)
fig.tight_layout(h_pad=0.4, w_pad=0.4)
speichern(fig, "umap_parameter.pdf")

fig, axes = plt.subplots(1, 3, figsize=(5.8, 1.95))
for ax, (seed, Ei) in zip(axes, [(0, E), (1, cache["seed1"]), (2, cache["seed2"])]):
    streubild(ax, Ei, study, s=0.5, beschriften=False, alpha=0.6)
    for t in ["PRAD", "LIHC", "BRCA"]:
        x_, y_ = np.median(Ei[study == t], axis=0)
        ax.annotate(t, (x_, y_), fontsize=6, ha="center", va="center", fontweight="semibold", color=TEXT,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))
    ohne_achsen(ax)
    ax.set_title(f"Zufallsstartwert {seed}", fontsize=6.8, fontweight="normal")
fig.tight_layout(w_pad=0.4)
speichern(fig, "umap_startwert.pdf")
print("fertig")
