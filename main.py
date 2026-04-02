import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn.decomposition import PCA

from sklearn.feature_selection import VarianceThreshold

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.ensemble import IsolationForest

from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score
)

FILE_PATH = r"C:\Users\nasty\Desktop\курсовая-диплом\силос-всякий.xlsx"

QUALITY_COL = "Кач-во корма"  # 1=отличн,2=хорош,3=плох
ID_LIKE_COLS = {"Год", "Регист номер", "Объем T"}
EXCLUDE_ID_LIKE_FROM_CLUSTERING = True

# parsing/cleaning
AUTOCONVERT_MIN_NON_NA_RATIO = 0.80
DROP_ROWS_WITH_ALL_FEATURES_NA = True
KEEP_ROWS_WITH_QUALITY_NA = True
# feature selection
VAR_THRESHOLD = 1e-6
CORR_DROP_THRESHOLD = 0.95

# preprocessing
USE_POWER_TRANSFORM = True
USE_ROBUST_SCALER = True
COMPARE_RAW_VS_PCA = True
PCA_VARIANCE_TARGET = 0.95

# outliers: "чувствительность"
RUN_OUTLIER_SENSITIVITY = True
OUTLIER_CONTAMINATIONS = [0.0, 0.05, 0.10]
RANDOM_STATE = 42

# clustering
K_RANGE = range(2, 11)
ALGORITHMS = ("kmeans", "ward")

# stability (KMeans)
STABILITY_SEEDS = [RANDOM_STATE + i for i in range(12)]
SUBSAMPLE_RUNS = 12
SUBSAMPLE_FRAC = 0.80

# feature variants
RUN_WITHOUT_MOISTURE = True
MOISTURE_COL = "Влага %"

# plots
SHOW_PLOTS = True
PLOT_HEATMAP = True
PLOT_METRICS_VS_K = True
PLOT_SILHOUETTE_FOR_BEST = True
SILHOUETTE_TOP_N = 3
HEATMAP_ANNOT = True
HEATMAP_FMT = ".2f"
HEATMAP_MAX_COLS_FOR_ANNOT = 35


def safe_int_quality(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").round().astype("Int64")


def parse_messy_number(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    s = str(x).strip()
    if s in {"", "-", "—", "–"}:
        return np.nan

    s = s.replace("\u00A0", " ").replace(" ", "")

    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    else:
        s = s.replace(",", ".")

    return pd.to_numeric(s, errors="coerce")


def auto_convert_numeric_like_text(df: pd.DataFrame, exclude_cols: set, min_non_na_ratio: float):
    out = df.copy()
    obj_cols = out.select_dtypes(include=["object", "category"]).columns.tolist()
    converted_cols = []
    for col in obj_cols:
        if col in exclude_cols:
            continue
        converted = out[col].map(parse_messy_number)
        ratio = float(converted.notna().mean())
        if ratio >= min_non_na_ratio:
            out[col] = converted.astype(float)
            converted_cols.append((col, ratio))
    return out, converted_cols


def prune_by_corr_verbose(df_feat: pd.DataFrame, threshold: float):
    corr = df_feat.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop, reasons = [], []
    for col in upper.columns:
        high = upper[col][upper[col] > threshold]
        if len(high) > 0:
            partner = high.idxmax()
            value = float(high.max())
            to_drop.append(col)
            reasons.append((col, partner, value))

    kept = [c for c in df_feat.columns if c not in to_drop]
    return kept, to_drop, reasons


def corr_heatmap_annot(df_num: pd.DataFrame, title: str):
    if df_num.shape[1] < 2:
        return
    corr = df_num.corr()
    annot = HEATMAP_ANNOT and (df_num.shape[1] <= HEATMAP_MAX_COLS_FOR_ANNOT)

    plt.figure(figsize=(18, 14))
    sns.heatmap(
        corr,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.3,
        annot=annot, fmt=HEATMAP_FMT,
        annot_kws={"fontsize": 7},
        cbar_kws={"shrink": 0.85},
    )
    plt.title(title + ("" if annot else " (too many cols for numbers)"))
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()


def eval_internal(X: np.ndarray, labels: np.ndarray) -> dict:
    out = {"sil": np.nan, "dbi": np.nan, "ch": np.nan}
    if len(np.unique(labels)) < 2:
        return out
    out["sil"] = float(silhouette_score(X, labels))
    out["dbi"] = float(davies_bouldin_score(X, labels))
    out["ch"] = float(calinski_harabasz_score(X, labels))
    return out


def eval_external_vs_quality(labels: np.ndarray, quality: pd.Series) -> dict:
    q = pd.to_numeric(quality, errors="coerce")
    mask = ~pd.isna(q)
    if mask.sum() == 0:
        return {"ARI": np.nan, "NMI": np.nan, "AMI": np.nan}

    y = q[mask].astype(int).values
    cl = pd.Series(labels, index=quality.index)[mask].values

    return {
        "ARI": float(adjusted_rand_score(y, cl)),
        "NMI": float(normalized_mutual_info_score(y, cl)),
        "AMI": float(adjusted_mutual_info_score(y, cl)),
    }


def fit_predict(alg: str, X: np.ndarray, k: int, random_state: int):
    if alg == "kmeans":
        model = KMeans(n_clusters=int(k), random_state=random_state, n_init="auto")
        labels = model.fit_predict(X)
        return labels, {"model": model}

    if alg == "ward":
        model = AgglomerativeClustering(n_clusters=int(k), linkage="ward")
        labels = model.fit_predict(X)
        return labels, {"model": model}

    raise ValueError(f"Unknown alg: {alg}")


def stability_seed_kmeans(X: np.ndarray, k: int, seeds: list[int]) -> float:
    labs = []
    for s in seeds:
        km = KMeans(n_clusters=int(k), random_state=int(s), n_init="auto")
        labs.append(km.fit_predict(X))

    vals = []
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            vals.append(adjusted_rand_score(labs[i], labs[j]))

    return float(np.mean(vals)) if vals else np.nan


def stability_subsample_kmeans(X: np.ndarray, k: int, runs: int, frac: float, seed0: int = 42) -> float:
    rng = np.random.default_rng(seed0)
    n = len(X)
    size = int(n * frac)

    idx_list = [rng.choice(n, size=size, replace=False) for _ in range(runs)]
    labs = []
    for i, idx in enumerate(idx_list):
        km = KMeans(n_clusters=int(k), random_state=int(seed0 + i), n_init="auto")
        labs.append((idx, km.fit_predict(X[idx])))

    vals = []
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            idx_i, lab_i = labs[i]
            idx_j, lab_j = labs[j]
            common = np.intersect1d(idx_i, idx_j)
            if len(common) < 30:
                continue
            li = pd.Series(lab_i, index=idx_i).loc[common].values
            lj = pd.Series(lab_j, index=idx_j).loc[common].values
            vals.append(adjusted_rand_score(li, lj))

    return float(np.mean(vals)) if vals else np.nan


def choose_best(metrics_df: pd.DataFrame) -> dict:

    def key(r):
        sil = -1e9 if pd.isna(r["sil"]) else float(r["sil"])
        stab_sub = -1e9 if pd.isna(r["stab_subsample_ARI"]) else float(r["stab_subsample_ARI"])
        stab_seed = -1e9 if pd.isna(r["stab_seed_ARI"]) else float(r["stab_seed_ARI"])
        dbi = 1e9 if pd.isna(r["dbi"]) else float(r["dbi"])
        mcs = -1e9 if pd.isna(r["min_cluster_size"]) else float(r["min_cluster_size"])
        return (sil, stab_sub, stab_seed, -dbi, mcs)

    best_i = max(metrics_df.index, key=lambda i: key(metrics_df.loc[i]))
    return metrics_df.loc[best_i].to_dict()


def plot_metrics_vs_k(metrics_df: pd.DataFrame, title_prefix: str):
    if metrics_df.empty:
        return
    dfp = metrics_df.copy()
    dfp["k"] = dfp["k"].astype(int)
    algs = sorted(dfp["alg"].unique().tolist())

    fig = plt.figure(figsize=(16, 10))

    ax1 = fig.add_subplot(2, 2, 1)
    for alg in algs:
        d = dfp[dfp["alg"] == alg].sort_values("k")
        ax1.plot(d["k"], d["sil"], marker="o", label=alg)
    ax1.set_title("Silhouette vs K (higher is better)")
    ax1.set_xlabel("K"); ax1.set_ylabel("Silhouette")
    ax1.grid(True, alpha=0.3); ax1.legend()

    ax2 = fig.add_subplot(2, 2, 2)
    for alg in algs:
        d = dfp[dfp["alg"] == alg].sort_values("k")
        ax2.plot(d["k"], d["dbi"], marker="o", label=alg)
    ax2.set_title("Davies–Bouldin vs K (lower is better)")
    ax2.set_xlabel("K"); ax2.set_ylabel("DBI")
    ax2.grid(True, alpha=0.3); ax2.legend()

    ax3 = fig.add_subplot(2, 2, 3)
    for alg in algs:
        d = dfp[dfp["alg"] == alg].sort_values("k")
        ax3.plot(d["k"], d["ch"], marker="o", label=alg)
    ax3.set_title("Calinski–Harabasz vs K (higher is better)")
    ax3.set_xlabel("K"); ax3.set_ylabel("CH")
    ax3.grid(True, alpha=0.3); ax3.legend()

    ax4 = fig.add_subplot(2, 2, 4)
    for alg in algs:
        d = dfp[dfp["alg"] == alg].sort_values("k")
        ax4.plot(d["k"], d["stab_subsample_ARI"], marker="o", label=alg)
    ax4.set_title("Stability (subsample ARI) vs K (KMeans only)")
    ax4.set_xlabel("K"); ax4.set_ylabel("mean ARI")
    ax4.grid(True, alpha=0.3); ax4.legend()

    fig.suptitle(f"{title_prefix}Metrics vs K", fontsize=14)
    fig.tight_layout()


def silhouette_plot_kmeans(X: np.ndarray, k_list, random_state: int, title_prefix: str):
    for n_clusters in k_list:
        if n_clusters >= len(X) or n_clusters < 2:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        ax1.set_xlim([-0.2, 1.0])
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        clusterer = KMeans(n_clusters=int(n_clusters), random_state=random_state, n_init="auto")
        cluster_labels = clusterer.fit_predict(X)

        silhouette_avg = silhouette_score(X, cluster_labels)
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            vals = sample_silhouette_values[cluster_labels == i]
            vals.sort()
            size_i = vals.shape[0]
            y_upper = y_lower + size_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                              facecolor=color, edgecolor=color, alpha=0.7)
            ax1.text(-0.05, y_lower + 0.5 * size_i, str(i))
            y_lower = y_upper + 10

        ax1.set_title("Silhouette plot (KMeans)")
        ax1.set_xlabel("Silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax1.set_yticks([])
        ax1.set_xticks([-0.2, -0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        pca2 = PCA(n_components=2, random_state=random_state)
        X_pca2 = pca2.fit_transform(X)
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X_pca2[:, 0], X_pca2[:, 1], marker=".", s=50, lw=0,
                    alpha=0.7, c=colors, edgecolor="k")

        centers = pca2.transform(clusterer.cluster_centers_)
        ax2.scatter(centers[:, 0], centers[:, 1], marker="o", c="white", alpha=1,
                    s=200, edgecolor="k")
        for i, c0 in enumerate(centers):
            ax2.scatter(c0[0], c0[1], marker=f"${i}$", alpha=1, s=60, edgecolor="k")

        ax2.set_title("PCA view of clusters")
        ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")

        fig.suptitle(f"{title_prefix}KMeans silhouette | k={n_clusters} | avg={silhouette_avg:.3f}", fontsize=14)
        fig.tight_layout()


def cluster_profile(df: pd.DataFrame, cluster_col: str, feat_cols: list[str]) -> pd.DataFrame:
    g = df.groupby(cluster_col)[feat_cols]
    med = g.median(numeric_only=True)
    q25 = g.quantile(0.25, numeric_only=True)
    q75 = g.quantile(0.75, numeric_only=True)
    out = pd.concat([med.add_suffix("_median"), (q75 - q25).add_suffix("_IQR")], axis=1)
    return out


sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 250)
pd.set_option("display.width", 240)

df = pd.read_excel(FILE_PATH)

if QUALITY_COL not in df.columns:
    raise KeyError(f"Не найден столбец качества: {QUALITY_COL}")

df[QUALITY_COL] = safe_int_quality(df[QUALITY_COL])

exclude_for_autoconv = set([c for c in ID_LIKE_COLS if c in df.columns])
df, converted_cols = auto_convert_numeric_like_text(
    df, exclude_cols=exclude_for_autoconv, min_non_na_ratio=AUTOCONVERT_MIN_NON_NA_RATIO
)
if converted_cols:
    print("\n[AutoConvert] Converted numeric-like text columns:")
    for c, r in converted_cols:
        print(f"  {c} (non-NA ratio after convert: {r:.2f})")

num_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
cols_exclude = {QUALITY_COL}
if EXCLUDE_ID_LIKE_FROM_CLUSTERING:
    cols_exclude |= set([c for c in ID_LIKE_COLS if c in df.columns])

feat_cols = [c for c in num_cols_all if c not in cols_exclude]
if len(feat_cols) == 0:
    raise ValueError("Не осталось числовых признаков для кластеризации.")

if DROP_ROWS_WITH_ALL_FEATURES_NA:
    before = len(df)
    df = df.dropna(subset=feat_cols, how="all").copy()
    print(f"\n[Cleaning] Dropped rows with all feature cols NA: {before - len(df)}")

if not KEEP_ROWS_WITH_QUALITY_NA:
    before = len(df)
    df = df.dropna(subset=[QUALITY_COL]).copy()
    print(f"\n[Cleaning] Dropped rows with {QUALITY_COL}=NA: {before - len(df)}")

X0 = df[feat_cols].copy()
X0 = X0.fillna(X0.median(numeric_only=True))

vt = VarianceThreshold(VAR_THRESHOLD)
X_vt = vt.fit_transform(X0.values)
mask_vt = np.asarray(vt.get_support(), dtype=bool)
kept_after_vt = list(np.array(feat_cols)[mask_vt])
print("\n[Feature selection] VarianceThreshold")
print("Kept:", len(kept_after_vt), "Dropped:", len(feat_cols) - len(kept_after_vt))

df_vt = pd.DataFrame(X_vt, columns=kept_after_vt, index=df.index)

keep_cols, dropped_corr, corr_reasons = prune_by_corr_verbose(df_vt, CORR_DROP_THRESHOLD)
print("\n[Feature selection] Correlation filter")
print("Kept:", len(keep_cols), "Dropped:", len(dropped_corr))
if dropped_corr:
    print("Dropped high-corr:", dropped_corr)
    print("Top reasons (dropped -> kept, corr):")
    for d, k, v in corr_reasons[:20]:
        print(f"  {d} -> {k}, corr={v:.3f}")

df_feat_base = df_vt[keep_cols].copy()

if SHOW_PLOTS and PLOT_HEATMAP:
    corr_heatmap_annot(df_feat_base, "Тепловая карта корреляций показателей химического состава кормов")

print("\n[Feature variants]")
variants = {"all": df_feat_base.copy()}
if RUN_WITHOUT_MOISTURE and MOISTURE_COL in df_feat_base.columns:
    variants["no_moisture"] = df_feat_base.drop(columns=[MOISTURE_COL]).copy()
for name, vdf in variants.items():
    print(f"  - {name}: n_features={vdf.shape[1]}")

def build_preprocess(use_pca: bool) -> Pipeline:
    steps = [("imp", SimpleImputer(strategy="median"))]
    if USE_POWER_TRANSFORM:
        steps.append(("pt", PowerTransformer(method="yeo-johnson", standardize=True)))
    if USE_ROBUST_SCALER:
        steps.append(("sc", RobustScaler()))
    if use_pca:
        steps.append(("pca", PCA(n_components=float(PCA_VARIANCE_TARGET), svd_solver="full", random_state=RANDOM_STATE)))
    return Pipeline(steps)

def get_inlier_mask(X_num: np.ndarray, contamination: float) -> np.ndarray:
    if contamination is None or contamination <= 0:
        return np.ones(X_num.shape[0], dtype=bool)
    if X_num.shape[0] < 20 or X_num.shape[1] < 2:
        return np.ones(X_num.shape[0], dtype=bool)
    iso = IsolationForest(
        n_estimators=500,
        contamination=float(contamination),
        random_state=RANDOM_STATE
    )
    return (iso.fit_predict(X_num) == 1)

def run_experiment(variant_name: str, df_feat_variant: pd.DataFrame, use_pca: bool, contamination: float) -> tuple[pd.DataFrame, dict]:
    prep = build_preprocess(use_pca=use_pca)

    X_all = df_feat_variant.values
    inlier = get_inlier_mask(X_all, contamination=contamination)
    idx_core = df_feat_variant.index[inlier]

    X_core = prep.fit_transform(df_feat_variant.loc[idx_core].values)

    rows = []
    for alg in ALGORITHMS:
        for k in K_RANGE:
            if k >= len(X_core):
                continue

            labels, _ = fit_predict(alg, X_core, k=int(k), random_state=RANDOM_STATE)
            ins = eval_internal(X_core, labels)
            ext = eval_external_vs_quality(labels, df.loc[idx_core, QUALITY_COL])

            row = {
                "variant": variant_name,
                "space": "PCA" if use_pca else "RAW",
                "outlier_cont": float(contamination),
                "alg": alg,
                "k": int(k),
                "n_core": int(len(idx_core)),
                "sil": ins["sil"],
                "dbi": ins["dbi"],
                "ch": ins["ch"],
                "ARI_vs_quality": ext["ARI"],
                "NMI_vs_quality": ext["NMI"],
                "AMI_vs_quality": ext["AMI"],
                "stab_seed_ARI": np.nan,
                "stab_subsample_ARI": np.nan,
                "min_cluster_size": int(np.min(np.bincount(labels))) if len(labels) else np.nan
            }

            if alg == "kmeans":
                row["stab_seed_ARI"] = stability_seed_kmeans(X_core, k=int(k), seeds=STABILITY_SEEDS)
                row["stab_subsample_ARI"] = stability_subsample_kmeans(
                    X_core, k=int(k), runs=SUBSAMPLE_RUNS, frac=SUBSAMPLE_FRAC, seed0=RANDOM_STATE
                )

            rows.append(row)

    metrics = pd.DataFrame(rows)
    best = choose_best(metrics)

    best["variant"] = variant_name
    best["use_pca"] = use_pca
    best["contamination"] = float(contamination)
    best["idx_core"] = idx_core
    best["inlier_mask"] = inlier
    best["prep"] = prep
    return metrics, best

conts = OUTLIER_CONTAMINATIONS if RUN_OUTLIER_SENSITIVITY else [OUTLIER_CONTAMINATIONS[-1]]
spaces = [False, True] if COMPARE_RAW_VS_PCA else [False]

all_metrics = []
best_candidates = []

for variant_name, df_feat_variant in variants.items():
    for use_pca in spaces:
        for cont in conts:
            metrics, best = run_experiment(
                variant_name=variant_name,
                df_feat_variant=df_feat_variant,
                use_pca=use_pca,
                contamination=float(cont)
            )
            all_metrics.append(metrics)
            best_candidates.append(best)

metrics_all = pd.concat(all_metrics, ignore_index=True)

best_global = None
def global_key(b):
    sil = -1e9 if pd.isna(b["sil"]) else float(b["sil"])
    stab_sub = -1e9 if pd.isna(b["stab_subsample_ARI"]) else float(b["stab_subsample_ARI"])
    stab_seed = -1e9 if pd.isna(b["stab_seed_ARI"]) else float(b["stab_seed_ARI"])
    dbi = 1e9 if pd.isna(b["dbi"]) else float(b["dbi"])
    mcs = -1e9 if pd.isna(b["min_cluster_size"]) else float(b["min_cluster_size"])
    # лёгкое предпочтение простоты: RAW чуть предпочтительнее при равных
    space_bonus = 0.0001 if b["space"] == "RAW" else 0.0
    # и предпочтение не выкидывать данные без необходимости
    core_bonus = 0.000001 * float(b["n_core"])
    return (sil + space_bonus + core_bonus, stab_sub, stab_seed, -dbi, mcs)

for b in best_candidates:
    if best_global is None or global_key(b) > global_key(best_global):
        best_global = b

print("\n==================== RESULT ====================")
print("Best overall:", {k: best_global[k] for k in [
    "variant","space","outlier_cont","alg","k","n_core","sil","dbi","ch",
    "stab_seed_ARI","stab_subsample_ARI",
    "ARI_vs_quality","NMI_vs_quality","AMI_vs_quality","min_cluster_size"
]})

variant_name = str(best_global["variant"])
use_pca = (best_global["space"] == "PCA")
contamination = float(best_global["outlier_cont"])
alg = str(best_global["alg"])
k_best = int(best_global["k"])

df_feat_final = variants[variant_name]
prep_final = build_preprocess(use_pca=use_pca)

inlier_final = get_inlier_mask(df_feat_final.values, contamination=contamination)
idx_core = df_feat_final.index[inlier_final]

X_core_final = prep_final.fit_transform(df_feat_final.loc[idx_core].values)
labels_core, _ = fit_predict(alg, X_core_final, k=k_best, random_state=RANDOM_STATE)

df_res = df.copy()
df_res["Outlier"] = False
df_res.loc[df_feat_final.index[~inlier_final], "Outlier"] = True

df_res["Cluster"] = -1
df_res.loc[idx_core, "Cluster"] = labels_core

print("\n--- Crosstab: Cluster vs Quality (core only) ---")
print(pd.crosstab(df_res.loc[idx_core, "Cluster"], df_res.loc[idx_core, QUALITY_COL]))

prof = cluster_profile(df_res.loc[idx_core], "Cluster", keep_cols).round(3)
print("\n--- Cluster profile (median + IQR) on selected features (core) ---")
print(prof.to_string())

print("\n=== TOP-5 per outlier_cont (best variant & best space) ===")
cols = ["variant","space","outlier_cont","alg","k","n_core","sil","dbi","ch",
        "stab_subsample_ARI","stab_seed_ARI","min_cluster_size"]

m = metrics_all[
    (metrics_all["variant"] == best_global["variant"]) &
    (metrics_all["space"] == best_global["space"])
].copy()

for cont in sorted(m["outlier_cont"].unique()):
    d = (m[m["outlier_cont"] == cont]
         .sort_values(["sil","stab_subsample_ARI","stab_seed_ARI"],
                      ascending=[False, False, False]))
    print("\n=== outlier_cont =", cont, "===")
    print(d[cols].head(5).to_string(index=False))

if SHOW_PLOTS and PLOT_METRICS_VS_K:
    df_plot = metrics_all[
        (metrics_all["variant"] == best_global["variant"]) &
        (metrics_all["space"] == best_global["space"]) &
        (metrics_all["outlier_cont"] == best_global["outlier_cont"])
    ].copy()
    plot_metrics_vs_k(
        df_plot,
        title_prefix=f"CORE | {best_global['variant']} | {best_global['space']} | outliers={best_global['outlier_cont']} | "
    )

if SHOW_PLOTS and PLOT_SILHOUETTE_FOR_BEST and alg == "kmeans":
    df_plot_km = metrics_all[
        (metrics_all["variant"] == best_global["variant"]) &
        (metrics_all["space"] == best_global["space"]) &
        (metrics_all["outlier_cont"] == best_global["outlier_cont"]) &
        (metrics_all["alg"] == "kmeans")
    ].copy()
    top_k = (
        df_plot_km.sort_values("sil", ascending=False)
                  .head(int(SILHOUETTE_TOP_N))["k"].astype(int).tolist()
    )
    top_k = sorted(set(top_k + [k_best]))
    silhouette_plot_kmeans(
        X_core_final,
        k_list=top_k,
        random_state=RANDOM_STATE,
        title_prefix=f"CORE | {best_global['variant']} | {best_global['space']} | outliers={best_global['outlier_cont']} | "
    )

if SHOW_PLOTS:
    try:
        plt.show()
    finally:
        plt.close('all')
