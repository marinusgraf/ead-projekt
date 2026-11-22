"""Abbildungen für die Zwischenpräsentation.

Aufruf aus dem Projektordner:  .venv/bin/python praesentation/abbildungen.py
Voraussetzung: notebook_aufbereitung.ipynb, notebook_pca.ipynb und notebook_umap.ipynb wurden einmal ausgeführt
(Binärdatei der Rohmatrix, aufbereitete Matrix, Hauptkomponenten, UMAP-Einbettung).
Die Rechnungen folgen den Notebooks; die zusätzlichen UMAP-Varianten werden beim ersten Lauf in
data/pancanatlas/praesentation_umap_varianten.npz zwischengespeichert.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
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
    p = Path("/usr/share/texmf-dist/fonts/opentype/public/fira") / f
    if p.exists():
        font_manager.fontManager.addfont(str(p))
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
nb_pca = np.load(DATA_DIR / "nachbarn_pca50.npy")
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
fig, ax = plt.subplots(figsize=(5.0, 1.95))
x = np.arange(1, len(v) + 1)
k90 = int(np.searchsorted(np.cumsum(v) / v.sum(), 0.9)) + 1                 # so viele Gene tragen 90 % der Varianz (wie D2)
ax.fill_between(x[:k90], v[:k90], color=BLAU, alpha=0.25, linewidth=0)
ax.plot(x, v, color=BLAU, lw=1.6)
ax.axvline(k90, color=MUTED, lw=0.8, ls=(0, (3, 2)))
ax.text(k90 + 150, v[0] * 0.8, f"{de(k90)} Gene behalten:\n{prozent(v[:k90].sum() / v.sum())} der Gesamtvarianz",
        fontsize=6.8, color=TEXT, va="top")
ax.set_xlabel("Gene, nach Varianz sortiert")
ax.set_ylabel(r"Varianz von $\log_2(x+1)$")
ax.xaxis.set_major_formatter(lambda v_, _: de(v_))
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

# ================================================================ 8. Protokoll der Aufbereitung
print("8 Protokoll")
prot = pd.DataFrame(PARAMETER["protokoll"])
SCHRITTE = {"alle Proben": "Rohdaten", "A1 nur Primärtumoren": "nur Primärtumoren", "A3 ohne Do_not_use": "ohne „nicht verwenden“",
            "A4 nur Illumina HiSeq": "nur Illumina HiSeq", "A2 eine Probe pro Patient": "eine Probe pro Patient",
            "B1 Gene mit fehlenden Werten entfernt": "ohne Gene mit fehlenden Werten", "D1 Expressionsfilter": "Expressionsfilter",
            "D3 Geschlechtsgene entfernt": "ohne Geschlechtsgene", "D2 Gene mit 90% der Varianz": "Gene mit 90 % der Varianz"}
prot["name"] = prot["Schritt"].map(SCHRITTE)
fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.35), sharey=True)
y = np.arange(len(prot))[::-1]
for ax, spalte, farbe in [(axes[0], "Proben", BLAU), (axes[1], "Gene", ORANGE)]:
    werte = prot[spalte].to_numpy()
    geaendert = np.r_[True, werte[1:] != werte[:-1]]
    ax.barh(y, werte, color=np.where(geaendert, farbe, HELLGRAU), height=0.66)
    for yi, w_, g in zip(y, werte, geaendert):
        ax.text(w_ + werte.max() * 0.015, yi, de(w_), va="center", fontsize=6.6, color=TEXT if g else MUTED)
    ax.set_xlim(0, werte.max() * 1.25)
    ax.set_xticks([])
    ax.set_title(spalte, loc="left")
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", length=0)
axes[0].set_yticks(y, prot["name"])
speichern(fig, "protokoll.pdf")

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

BESCHRIFTUNG = {
    "pca_12.pdf": ["LGG", "GBM", "PCPG", "ACC", "UVM", "KIRC", "KICH", "LAML", "LIHC", "PAAD", "LUSC", "HNSC", "CESC", "COAD", "READ", "BRCA", "PRAD", "SARC"],
    "pca_34.pdf": ["LAML", "DLBC", "KIRC", "KIRP", "PAAD", "PRAD", "THCA", "UVM", "SKCM", "CESC", "HNSC", "THYM", "BRCA", "LIHC"],
}
for (i, j), name in [((0, 1), "pca_12.pdf"), ((2, 3), "pca_34.pdf")]:
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    streubild(ax, Z[:, [i, j]], study, s=2.6, alpha=0.5, nur=BESCHRIFTUNG[name])
    ax.set_xlabel(f"Hauptkomponente {i + 1} ({prozent(ev[i])})")
    ax.set_ylabel(f"Hauptkomponente {j + 1} ({prozent(ev[j])})")
    ax.grid(axis="both")
    legende_gruppen(ax, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=6.3)
    ax.tick_params(labelsize=6.5)
    speichern(fig, name)

# Ladungen: Gene mit den größten Gewichten an beiden Enden
PROGRAMME = {0: ("Drüsenepithel", "Nervengewebe"), 1: ("Leber, Niere", "Plattenepithel"),      # (Ende −, Ende +)
             2: ("Niere, Drüsen", "Schleimhaut"), 3: ("Drüsenepithel", "Immunzellen")}
fig, axes = plt.subplots(1, 4, figsize=(5.9, 2.35))
for j, ax in enumerate(axes):
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
speichern(fig, "pca_ladungen.pdf")

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
lang = 15
ax.annotate("", xy=richtung * lang, xytext=-richtung * 4, arrowprops=dict(arrowstyle="-|>", color=BLAU, lw=1.8))
ax.text(*(richtung * lang + np.array([-0.5, 1.2])), f"eine Achse:\n{prozent(anteil1)} der Streuung", fontsize=6.6, color=BLAU,
        ha="right", va="bottom", fontweight="semibold")
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

# Nachbarschaft: Anteil der 15 nächsten Nachbarn (50 Hauptkomponenten) pro Paar von Tumortypen
TYPEN = [t for typen in [GRUPPEN[g][1] for g in GRUPPEN] for t in typen]
paare = pd.crosstab(np.repeat(study, nb_pca.shape[1]), study[nb_pca.ravel()], normalize="index").loc[TYPEN, TYPEN]
eigen = pd.Series((study[nb_pca] == study[:, None]).mean(axis=1)).groupby(study).mean().sort_values()
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
CACHE = DATA_DIR / "praesentation_umap_varianten.npz"
varianten = [(k, d) for k in [5, 15, 50] for d in [0.1, 0.5]]
if CACHE.exists():
    cache = dict(np.load(CACHE))
else:
    import umap
    Y = Z[:, :50]
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
