import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import (
    KMeans, AgglomerativeClustering, DBSCAN,

)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score,
    davies_bouldin_score,

)
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from kneed import KneeLocator
import umap
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.preprocessing import StandardScaler
import os
import warnings
from sklearn.decomposition import IncrementalPCA
from hdbscan import HDBSCAN
from scipy.stats import kruskal, mannwhitneyu, shapiro, ttest_rel, wilcoxon, probplot
from collections import defaultdict
from scipy.cluster.hierarchy import dendrogram, linkage
import colorcet as cc
from scipy.stats import fisher_exact
from sklearn.preprocessing import LabelEncoder



warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"sklearn\.utils\.deprecation"
)
warnings.filterwarnings("ignore", message="n_jobs value 1 overridden to 1 by setting random_state")


pd.set_option('display.max_columns',None)

FIGURES_DIR = "testing_anomalies_final"
os.makedirs(FIGURES_DIR, exist_ok=True)
DIMS_REDUCE = list(range(2, 11)) + [50, 100, 150, 200]
KS = list(range(2, 15)) + list(range(20, 50, 4))
CLUSTER_ALGS = ['kmeans','kmeans_elbow', 'gmm', 'dbscan', "gmm_bic", 'hdbscan', 'hierarchical']
METRICS = ['silhouette', 'ch', 'db', 'mi', 'combined', 'combined_custom']



DATA_PATH = "EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena"
COMPRESSION = None
CHUNK_SIZE = 1000
VAR_PERC = 10
PCA_TARGET = 0.90




def compute_gene_variances(path, compression=None, chunksize=CHUNK_SIZE):
    var_list = []
    reader = pd.read_csv(
        path,
        sep="\t",
        index_col=0,
        compression=compression,
        chunksize=chunksize
    )
    for chunk in reader:
        chunk = chunk.astype(float).dropna(axis=0, how='any')
        var_list.append(chunk.var(axis=1, ddof=1))
    return pd.concat(var_list)



def filter_top_genes(path, compression, top_genes, chunksize=CHUNK_SIZE):
    dfs = []
    reader = pd.read_csv(
        path,
        sep="\t",
        index_col=0,
        compression=compression,
        chunksize=chunksize
    )
    for chunk in reader:
        chunk = chunk.astype(float).dropna(axis=0, how='any')
        dfs.append(chunk.loc[chunk.index.intersection(top_genes)])
    return pd.concat(dfs)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────


def prepare_tcga_pancan_old():
    print("Loading from:", DATA_PATH, "(compression =", COMPRESSION, ")")

    # 1) Compute variances
    gene_vars = compute_gene_variances(DATA_PATH, COMPRESSION)
    thresh = np.percentile(gene_vars, 100 - VAR_PERC)
    top_genes = gene_vars[gene_vars >= thresh].index
    print(f"→ Kept {len(top_genes)} genes (top {VAR_PERC}%)  "
          f"with var ≥ {thresh:.4f}")

    # 2) Load only top-variable genes
    df_top = filter_top_genes(DATA_PATH, COMPRESSION, top_genes)

    # 3) Log-transform (important for RNAseq)

    # 4) Transpose → samples × genes
    X = df_top.T



    print("X shape (samples × genes):", X.shape)

    # 5) Standardize
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    non_const = X_std.var(axis=0) > 1e-12
    X_std = X_std[:, non_const]
    # 6) Incremental PCA (fitting in batches)
    # Fit initial PCA
    n_features = X_std.shape[1]
    ipca = IncrementalPCA(n_components=n_features, batch_size=256)
    for batch_start in range(0, X_std.shape[0], 256):
        batch_end = batch_start + 256
        ipca.partial_fit(X_std[batch_start:batch_end])

    # Compute cumulative variance
    cumvar = np.cumsum(ipca.explained_variance_ratio_)

    # Find number of components to reach PCA_TARGET
    n_comp = np.searchsorted(cumvar, PCA_TARGET) + 1
    if n_comp > len(cumvar):
        n_comp = len(cumvar)

    print(f"→ Keeping {n_comp} components for {cumvar[n_comp - 1]:.2%} explained variance")

    # Re-fit PCA with correct number of components
    ipca = IncrementalPCA(n_components=n_comp, batch_size=256)
    X_pca = ipca.fit_transform(X_std)
    X_pca_df = pd.DataFrame(X_pca, index=X.index)
    print("Final PCA shape:", X_pca_df.shape)

    return X_pca_df



def prepare_tcga_pancan():
    print("Loading from:", DATA_PATH, "(compression =", COMPRESSION, ")")

    # 1) Compute per-gene variances
    gene_vars = compute_gene_variances(DATA_PATH, COMPRESSION)
    thresh = np.percentile(gene_vars, 100 - VAR_PERC)
    top_genes = gene_vars[gene_vars >= thresh].index
    print(f"→ Kept {len(top_genes)} genes (top {VAR_PERC}%) with var ≥ {thresh:.4f}")

    # 2) Load only top-variable genes
    df_top = filter_top_genes(DATA_PATH, COMPRESSION, top_genes)

    # 3) Transpose → samples × genes
    X = df_top.T
    print("X shape (samples × genes):", X.shape)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    non_const = X_std.var(axis=0) > 1e-12
    X_std = X_std[:, non_const]
    print(f"→ Non-constant genes retained: {X_std.shape[1]}")

    # 5) PCA with enough components for PCA_TARGET variance
    pca = PCA(n_components=PCA_TARGET, svd_solver='full')
    X_pca = pca.fit_transform(X_std)
    X_pca_df = pd.DataFrame(X_pca, index=X.index)
    print(f"→ Final PCA shape: {X_pca_df.shape} for {PCA_TARGET:.0%} variance explained")

    return X_pca_df



def merge_labels(X_reduced):

    clinical_path ="TCGA_phenotype_denseDataOnlyDownload.tsv"
    clinical = pd.read_csv(clinical_path, sep='\t', index_col=0)
    print(clinical.columns)
    clinical.index = clinical.index.str.strip()

    merged = X_reduced.join(clinical, how='inner')

    print(f"✅ Successfully merged: {merged.shape[0]} samples, {merged.shape[1]} columns.")
    return merged


def reduce_dims(X, method, n_components):
    if method == 'pca':
        return PCA(n_components=n_components, random_state=42).fit_transform(X)
    elif method == 'umap':
        return umap.UMAP(n_components=n_components,
                         n_neighbors= 10,
                         random_state=42,
                         n_epochs = 150,
                         low_memory=True).fit_transform(X)
    elif method == 'lle':
        return LocallyLinearEmbedding(n_components=n_components, n_neighbors=15, random_state=42).fit_transform(X)


def pick_kmeans_dim_k(X, dims, ks):
    mean_inertia = {}
    random_state = 42
    for d in dims:
        Xd = reduce_dims(X, 'umap',n_components=d)   # <-- no scaler here
        inertias = []
        for k in ks:
            km = KMeans(k, random_state=random_state).fit(Xd)
            inertias.append(km.inertia_)
        mean_inertia[d] = np.mean(inertias)

    best_dim = min(mean_inertia, key=mean_inertia.get)

    # 2. elbow 50 % on that dim ----------------------------------------
    X_best = reduce_dims(X, method = 'umap', n_components=best_dim)
    inertias = [KMeans(k, n_init=10, random_state=random_state).fit(X_best).inertia_
                for k in ks]

    thresh   = inertias[0] * 0.50
    k_elbow  = next((k for k, inn in zip(ks, inertias) if inn <= thresh), ks[-1])

    print(f"🏅 K-Means selector → dim = {best_dim}, k = {k_elbow}")
    return best_dim, k_elbow




def find_best_k_gmm(X, k_range):
    bics = []
    X = reduce_dims(X, method='pca', n_components=30)
    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(X)
        bics.append(gmm.bic(X))
    best_k = k_range[np.argmin(bics)]
    print(f"✅ GMM BIC suggests best k = {best_k}")
    return best_k


def tune_dbscan_eps(X, dim):
    k = 5
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    dist, _ = nbrs.kneighbors(X)
    k_dist = np.sort(dist[:, k-1])
    plt.figure(figsize=(8,5)); plt.plot(k_dist)
    #plt.title("DBSCAN k-distance curve"); plt.grid(True)
    plt.savefig(os.path.join(FIGURES_DIR, f"dbscan_k_distance_dim{dim}.png"), dpi=300)
    plt.close()
    knee = KneeLocator(range(len(k_dist)), k_dist, curve="convex", direction="increasing")
    return k_dist[knee.knee] if knee.knee is not None else np.percentile(k_dist, 90)


def combined_internal_score_geometric(sil, ch, db):
    ch_max = 15000
    db_max = 2.0
    ch_norm = min(ch / ch_max, 1.0)
    db_norm = min(db / db_max, 1.0)
    sil = max(sil, 1e-4)
    ch_norm = max(ch_norm, 1e-4)
    db_term = 1 / (1 + db_norm)

    return (sil * ch_norm * db_term) ** (1/3)




def metric_dict():
    return {alg: {} for alg in CLUSTER_ALGS}

def store(res, metric, alg, key, value):
    res[metric][alg][key] = value

# 🔽 INSERT HERE
def compute_custom_combined_metric(results):
    sil_all = results['silhouette']
    ch_all  = results['ch']
    db_all  = results['db']
    combined = {}

    for alg in sil_all:
        sil_vals = sil_all[alg]
        ch_vals  = ch_all.get(alg, {})
        db_vals  = db_all.get(alg, {})

        ch_values = np.array(list(ch_vals.values()))
        db_values = np.array(list(db_vals.values()))

        if len(ch_values) == 0 or len(db_values) == 0:
            continue

        ch_min, ch_max = ch_values.min(), ch_values.max()
        db_min, db_max = db_values.min(), db_values.max()

        combined[alg] = {}
        for key in sil_vals:
            sil = sil_vals.get(key, -1)
            ch = ch_vals.get(key, -1)
            db = db_vals.get(key, -1)

            ch_norm = (ch - ch_min) / (ch_max - ch_min + 1e-8)
            db_norm = (db - db_min) / (db_max - db_min + 1e-8)

            combined_score = 0.8 * sil - 0.2 * db_norm
            combined[alg][key] = combined_score

    results['combined_custom'] = combined



def scores(X,lbl, cancer_truth):
    if len(set(lbl))<2:
        return -1,-1,-1, -1, -1
    sil = silhouette_score(X,lbl)
    ch = calinski_harabasz_score(X,lbl)
    db =davies_bouldin_score(X,lbl)
    mi = adjusted_mutual_info_score(cancer_truth, lbl)
    n_score = combined_internal_score_geometric(sil, ch, db)

    return (sil, ch, db, mi, n_score)


def get_distinct_colors(n):
    """
    Generate n visually distinct colors using matplotlib's tab20, tab20b, and tab20c.
    """
    base = plt.get_cmap('tab20').colors + plt.get_cmap('tab20b').colors + plt.get_cmap('tab20c').colors
    if n <= len(base):
        return base[:n]
    else:
        # Repeat colors if more than available, but ideally not recommended
        print("⚠ Warning: More clusters than distinct colors. Colors may repeat.")
        return base * ((n // len(base)) + 1)



def bootstrap_scores(X, y_true, algo, kw, n, seed):
    alpha = 0.5
    rng = np.random.RandomState(seed)
    sil_all, mi_all, combined_all = [], [], []
    for _ in range(n):
        idx= rng.choice(len(X),len(X),True)
        Xb = X[idx]
        yb = y_true.iloc[idx]
        labels = algo(**kw).fit_predict(X[idx])
        if len(set(labels))>1:
            sil = silhouette_score(Xb, labels)
            mi = adjusted_mutual_info_score(yb, labels)
            combined =alpha*sil + (1-alpha)*mi
            sil_all.append(sil)
            mi_all.append(mi)
            combined_all.append(combined)
    return {'silhouette': np.array(sil_all),
        'mi': np.array(mi_all),
        'combined': np.array(combined_all)}


def run_all(X_matched, X_train, X_test, best_k_km, best_dim_km, k_bic, cancer_type_train, cancer_type_test, n_boot, seed):
    X_h = X_train.copy()
    results = {m: metric_dict() for m in METRICS}
    all_cluster_labels = {}
    boot_sil = {}
    boot_combined = {}
    boot_mi = {}
    eps_dim = {}          # tuned ε  per PCA dim               (DBSCAN)
    best_combo = {}          # chosen (dim,k,mode) per algorithm


    #Hier
    if 'hierarchical' in CLUSTER_ALGS:
        for k in KS:
            lbl = AgglomerativeClustering(n_clusters=k).fit_predict(X_h)
            sil, ch, db, mi, comb = scores(X_h, lbl, cancer_type_train)
            key = (256, k, 'grid')
            store(results, 'silhouette', 'hierarchical', key, sil)
            store(results, 'ch', 'hierarchical', key, ch)
            store(results, 'db', 'hierarchical', key, db)
            store(results, 'mi', 'hierarchical', key, mi)
            store(results, 'combined', 'hierarchical', key, comb)
    for dim in DIMS_REDUCE:
        Xd = reduce_dims(X_train, 'umap', n_components=dim)
        eps = tune_dbscan_eps(Xd, dim)
        eps_dim[dim] = eps
        ms = 5

        # ========================================
        #  GRID  +  “FIXED”  BRANCHES
        # ========================================
        for alg in CLUSTER_ALGS:
            if alg == 'kmeans':
                for k in KS:  # KS = range(2,12)
                    lbl = KMeans(k, random_state=42).fit_predict(Xd)
                    sil, ch, db, mi, comb = scores(Xd, lbl, cancer_type_train)
                    key = (dim, k, 'grid')
                    store(results, 'silhouette', alg, key, sil)
                    store(results, 'ch', alg, key, ch)
                    store(results, 'db', alg, key, db)
                    store(results, 'mi', alg, key, mi)
                    store(results, 'combined', alg, key, comb)
                print("0")
            if alg == 'kmeans_elbow':
                if dim == best_dim_km:
                    lbl = KMeans(best_k_km, random_state=42).fit_predict(Xd)
                    sil, ch, db, mi, comb = scores(Xd, lbl, cancer_type_train)
                    key = (dim, best_k_km, 'best')
                    store(results, 'silhouette', alg, key, sil)
                    store(results, 'ch', alg, key, ch)
                    store(results, 'db', alg, key, db)
                    store(results, 'mi', alg, key, mi)
                    store(results, 'combined', alg, key, comb)


                print("1")
            elif alg == 'gmm':
                for mode, kset in [('fixed',[k_bic]), ('grid', KS)]:
                    for k in kset:
                        lbl = GaussianMixture(k, random_state=42).fit(Xd).predict(Xd)
                        sil, ch, db, mi, comb = scores(Xd, lbl, cancer_type_train)
                        key = (dim, k, mode)
                        store(results,'silhouette', alg, key, sil)
                        store(results,'ch',          alg, key, ch)
                        store(results,'db',          alg, key, db)
                        store(results, 'mi', alg, key, mi)
                        store(results, 'combined', alg, key, comb)
                print("2")

            elif alg == 'dbscan':
                lbl = DBSCAN(eps=eps, min_samples=ms).fit_predict(Xd)
                sil, ch, db, mi, comb = scores(Xd, lbl, cancer_type_train)
                key = (dim, None, 'auto')
                store(results,'silhouette', alg, key, sil)
                store(results,'ch',          alg, key, ch)
                store(results,'db',          alg, key, db)
                store(results, 'mi', alg, key, mi)
                store(results, 'combined', alg, key, comb)
                print("3")

            elif alg == 'gmm_bic':  # << NEW
                lbl = (GaussianMixture(k_bic, random_state=42)
                       .fit(Xd).predict(Xd))
                sil, ch, db, mi, comb = scores(Xd, lbl, cancer_type_train)
                key = (dim, k_bic, 'bic')
                store(results, 'silhouette', alg, key, sil)
                store(results, 'ch', alg, key, ch)
                store(results, 'db', alg, key, db)
                store(results, 'mi', alg, key, mi)
                store(results, 'combined', alg, key, comb)
                print("4")
            elif alg == 'hdbscan':
                clusterer = HDBSCAN(min_cluster_size=20, min_samples=10, prediction_data=True)
                lbl = clusterer.fit_predict(Xd)
                sil, ch, db, mi, comb = scores(Xd, lbl, cancer_type_train)
                key = (dim, None, 'auto')
                store(results, 'silhouette', alg, key, sil)
                store(results, 'ch', alg, key, ch)
                store(results, 'db', alg, key, db)
                store(results, 'mi', alg, key, mi)
                store(results, 'combined', alg, key, comb)
                print("5")
        print(f"✔ finished dim={dim}")

    compute_custom_combined_metric(results)


    for alg, table in results['combined_custom'].items():
        best_triplet = max(table.items(), key=lambda kv: kv[1])[0]
        best_combo[alg] = best_triplet
        # Bootstrap around that choice
        dim, k_val, mode = best_triplet
        Xd_best = reduce_dims(X_test, 'umap', dim)
        algo = None
        kw = None
        if alg == 'kmeans':
            algo = KMeans;   kw = {'n_clusters':int(k_val), 'random_state': 42}
        elif alg == 'kmeans_elbow':
            algo = KMeans; kw = {'n_clusters': int(k_val), 'random_state': 42}
        elif alg == 'gmm':
            algo = GaussianMixture; kw = {'n_components':int(k_val), 'random_state':42}
        elif alg == 'dbscan':
            algo = DBSCAN;   kw = {'eps':eps_dim[dim], 'min_samples': ms}
        elif alg == 'gmm_bic':
            algo = GaussianMixture
            kw = {'n_components': k_bic, 'random_state': 42}
        elif alg == 'hdbscan':
            algo = HDBSCAN
            kw = {'min_cluster_size': 20, 'min_samples': 10, 'prediction_data': True}
        else:
            continue

        boot = bootstrap_scores(Xd_best, cancer_type_test, algo, kw, n=n_boot, seed=seed)
        boot_sil[alg] = boot['silhouette']
        boot_combined[alg] = boot['combined']
        boot_mi[alg] = boot['mi']
        plt.boxplot(boot['combined'])
        plt.savefig(f'{FIGURES_DIR}/{alg}_combined_boot.png')
        plt.close()

    lbl = None
    for alg in CLUSTER_ALGS:
        dim, k, mode  = best_combo.get(alg)
        if alg == 'hierarchical':
            lbl = AgglomerativeClustering(n_clusters=k).fit_predict(X_matched)
            all_cluster_labels[alg] = pd.Series(lbl, index=X_matched.index)
            continue
        X_f = reduce_dims(X_matched, 'umap', dim)
        if alg == 'kmeans':
            lbl = KMeans(k, random_state=42).fit_predict(X_f)
        elif alg == 'kmeans_elbow':
            lbl = KMeans(k, random_state=42).fit_predict(X_f)
        elif alg == 'gmm':
            lbl = GaussianMixture(k, random_state=42).fit(X_f).predict(X_f)
        elif alg == 'dbscan':
            eps = eps_dim[dim]
            lbl = DBSCAN(eps=eps, min_samples=ms).fit_predict(X_f)
        elif alg == 'gmm_bic':  # << NEW
            lbl = GaussianMixture(k, random_state=42).fit(X_f).predict(X_f)
        elif alg == 'hdbscan':
            clusterer = HDBSCAN(min_cluster_size=20, min_samples=10, prediction_data=True)
            lbl = clusterer.fit_predict(X_f)
        all_cluster_labels[alg] = pd.Series(lbl, index=X_matched.index)

    acl = pd.DataFrame(all_cluster_labels)
    acl.to_csv("all_clusters_labels.csv", index=True)

    return results, boot_sil, boot_mi, boot_combined, best_combo, eps_dim, all_cluster_labels



def plot_heatmaps(results):

    for metric in METRICS:
        for alg in CLUSTER_ALGS:
            vals = results[metric][alg]
            if not vals:                       # nothing stored for this pair
                continue

            # ---- rows = dims,  columns = k’s --------------------------------
            dims = sorted({t[0] for t in vals})                 # first field
            ks = sorted({t[1] for t in vals if t[1] is not None})

            # allocate matrix filled with NaNs
            if alg in ('dbscan', 'hdbscan'):                     # only “auto”
                mat = np.full((len(dims), 1), np.nan)
                xlabels = ['auto']
                for i, d in enumerate(dims):
                    mat[i, 0] = vals.get((d, None, 'auto'), np.nan)

            else:                                                # regular algs
                mat = np.full((len(dims), len(ks)), np.nan)
                xlabels = ks
                for i, d in enumerate(dims):
                    for j, k in enumerate(ks):
                        # try grid first, fall back to fixed (for kmeans / gmm)
                        for mode in ('grid', 'best', 'bic'):
                            if (d, k, mode) in vals:
                                mat[i, j] = vals[(d, k, mode)]
                                break   # stop at first match

            # ---- draw --------------------------------------------------------
            plt.figure(figsize=(8, 6))
            sns.heatmap(mat,
                        xticklabels=xlabels,
                        yticklabels=dims,
                        annot=False, fmt=".4f",
                        cmap="viridis",
                        mask=np.isnan(mat))
            #plt.title(f"{alg.upper()} – {metric.upper()}  (UMAP)")
            plt.xlabel("k");   plt.ylabel("UMAP dim")
            plt.tight_layout()

            fname = f"{FIGURES_DIR}/{alg}_{metric}_heatmap.png"
            plt.savefig(fname, dpi=300)
            plt.close()
            print(f"✓ saved {fname}")



def print_best_params(results):
    print("\n🏆 Best parameter combos by metric:")
    for metric in METRICS:
        print(f"\n→ {metric.upper()}")
        for alg in CLUSTER_ALGS:
            vals = results[metric][alg]
            if not vals: continue
            key, score = (
                min(vals.items(), key=lambda kv: kv[1])
                if metric == "db"
                else max(vals.items(), key=lambda kv: kv[1])
            )

            dim, k, mode = key
            k_label = "auto" if k is None else k
            print(f"  {alg:<10} dim={dim:<3} k={k_label:<5} mode={mode:<6} "
                  f"score={score:.4f}")


def save_best_params_to_csv(results):
    output_path = "best_params.csv"
    rows = []
    for metric in METRICS:
        for alg in CLUSTER_ALGS:
            vals = results[metric][alg]
            if not vals:
                continue
            key, score = (
                min(vals.items(), key=lambda kv: kv[1])
                if metric == "db"
                else max(vals.items(), key=lambda kv: kv[1])
            )
            dim, k, mode = key
            k_label = "auto" if k is None else k
            rows.append({
                "algorithm": alg,
                "metric": metric,
                "dim": dim,
                "k": k_label,
                "mode": mode,
                "score": round(score, 4)
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"✅ Best parameters saved to {output_path}")


def get_glasbey_palette(n_colors):
    return cc.glasbey[:n_colors]

def visualize_clusters_tsne(X_scaled, results, eps_per_dim, method):
    X_h = X_scaled.copy()
    n_min_samp = 5
    for alg in CLUSTER_ALGS:
        cand = {key: val for key, val in results['combined_custom'][alg].items()
                if val >= 0}
        if not cand:
            continue

        best_key, _ = max(cand.items(), key=lambda kv: kv[1])
        best_dim, best_k, best_mode = best_key
        lbl = []
        if alg == 'hierarchical':
            lbl = AgglomerativeClustering(n_clusters=best_k).fit_predict(X_h)
            print(f"best dim hier: {best_dim}")
        # Step 1: reduce
        X_red = reduce_dims(X_scaled, method=method, n_components=best_dim)
        # Step 2: cluster
        if alg in ('kmeans', 'kmeans_elbow'):
            lbl = KMeans(n_clusters=best_k, random_state=42).fit_predict(X_red)
        elif alg in ('gmm', 'gmm_bic'):
            lbl = GaussianMixture(n_components=best_k, random_state=42).fit(X_red).predict(X_red)
        elif alg == 'dbscan':
            eps = eps_per_dim[best_dim]
            lbl = DBSCAN(eps=eps, min_samples=n_min_samp).fit_predict(X_red)
            n_clusters = len(set(lbl)) - (1 if -1 in lbl else 0)
            print(f"DBSCAN found ({n_clusters} clusters)")
        elif alg == 'hdbscan':
            clusterer = HDBSCAN(min_cluster_size=20, min_samples=10, prediction_data=True)
            lbl = clusterer.fit_predict(X_red)
            n_clusters = len(set(lbl)) - (1 if -1 in lbl else 0)
            print(f"HDBSCAN found ({n_clusters} clusters)")

        unique_labels = sorted(set(lbl))
        n_colors_needed = len(unique_labels)
        palete = get_glasbey_palette(n_colors_needed)

        # Map labels to palette index (e.g., -1 → 0, 0 → 1, etc.)
        label_to_color = {label: palete[i] for i, label in enumerate(unique_labels)}

        X_tsne = None
        if alg == 'hierarchical':
            X_tsne = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_h)
        else:
            X_tsne = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_red)


        # Step 4: plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=lbl, palette=label_to_color, s=25, legend=False)
        #plt.title(f"{alg.upper()} found {n_colors_needed} clusters (2D t-SNE on {method.upper()}-{best_dim}D)")
        plt.xlabel("t-SNE1"); plt.ylabel("t-SNE2")
        plt.tight_layout()
        fname = f"{alg}_{method}_tsne_clusters.png"
        plt.savefig(os.path.join(FIGURES_DIR, fname), dpi=300)
        plt.close()

        print(f"✅ Saved t-SNE cluster plot for {alg.upper()} ({method.upper()}, dim={best_dim})")


def plot_dendrogram(X_matched):
    method = 'ward'
    Z = linkage(X_matched, method=method)
    plt.figure(figsize=(12, 6))
    dendrogram(Z, no_labels=True)
    plt.tight_layout()
    plt.savefig("dendrogram.png", dpi=300)
    plt.close()





def print_bootstrap_summary(boot_dict, type):

    decimals = 4
    if type == 'sil':
        print("\n📊  Silhouette bootstrap summaries")
    elif type == 'mi':
        print("\n📊  MI bootstrap summaries")
    elif type == 'comb':
        print("\n📊  Combined bootstrap summaries")
    hdr = f"{'alg':12}  {'mean':>8}  {'std':>8}  {'2.5%':>8}  {'97.5%':>8}  n"
    print(hdr)
    print("-" * len(hdr))
    for name, arr in boot_dict.items():
        if len(arr) == 0:          # e.g. DBSCAN collapsed to one cluster
            print(f"{name:12}  <degenerate>")
            continue
        mean = np.round(arr.mean(),    decimals)
        std  = np.round(arr.std(ddof=1),decimals)
        p25  = np.round(np.percentile(arr,  2.5),decimals)
        p975 = np.round(np.percentile(arr, 97.5),decimals)
        print(f"{name:12}  {mean:8}  {std:8}  {p25:8}  {p975:8}  {len(arr)}")


def save_bootstrap_summary_to_csv(boot_dict, type):
    output_path = ""
    if type == 'sil':
        output_path = "silhouette_bootstrap_summary.csv"
    elif type == 'mi':
        output_path = "mi_bootstrap_summary.csv"
    elif type == "comb":
        output_path = "combined_bootstrap_summary.csv"
    decimals = 4
    rows = []
    for name, arr in boot_dict.items():
        if len(arr) == 0:
            continue
        mean = np.round(arr.mean(), decimals)
        std  = np.round(arr.std(ddof=1), decimals)
        p25  = np.round(np.percentile(arr, 2.5), decimals)
        p975 = np.round(np.percentile(arr, 97.5), decimals)

        rows.append({
            "algorithm": name,
            "mean": mean,
            "std_dev": std,
            "percentile_2.5": p25,
            "percentile_97.5": p975,
            "n_bootstraps": len(arr)
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"✅ Bootstrap summary saved to {output_path}")


def save_raw_bootstrap_to_csv(boot_dict, type):
    filename = ""
    if type == 'sil':
        filename = "silhouette_bootstrap_full.csv"
    elif type == 'mi':
        filename = "mi_bootstrap_full.csv"
    elif type == "comb":
        filename = "combined_bootstrap_full.csv"
    rows = []
    for alg, values in boot_dict.items():
        for i, score in enumerate(values):
            rows.append({"algorithm": alg, "replicate": i, "score": score})
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"✅ Raw bootstrap data saved to {filename}")



def plot_silhouette_bar(boot_dict):
    # Prepare means and stds
    means = {alg: vals.mean() for alg, vals in boot_dict.items()}
    stds  = {alg: vals.std()  for alg, vals in boot_dict.items()}

    # Create DataFrame
    df = pd.DataFrame({
        'Algorithm': list(means.keys()),
        'sil': list(means.values()),
        'Error': list(stds.values())
    })

    # Sort for consistent plotting
    df = df.sort_values("sil", ascending=False).reset_index(drop=True)
    colors = sns.color_palette("pastel", len(df))

    # Plot using matplotlib to add error bars manually
    plt.figure(figsize=(7, 5))
    x = np.arange(len(df))
    bars = plt.bar(x, df["sil"], color=colors, edgecolor="black")

    # Add error bars manually

    plt.errorbar(x, df["sil"], yerr=df["Error"], fmt='none', ecolor='black', capsize=4)

    plt.xticks(ticks=x, labels=df["Algorithm"], rotation=0)
    plt.ylabel("Silhouette Score")

    plt.tight_layout()
    plt.savefig("plot_A_silhouette_bars.png", dpi=300)
    plt.close()





def plot_mi_bar(boot_dict):
    # Prepare means and stds
    means = {alg: vals.mean() for alg, vals in boot_dict.items()}
    stds  = {alg: vals.std(ddof=1) for alg, vals in boot_dict.items()}

    # Create DataFrame
    df = pd.DataFrame({
        'Algorithm': list(means.keys()),
        'MI': list(means.values()),
        'Error': list(stds.values())
    })

    # Sort for consistent plotting
    df = df.sort_values("MI", ascending=False).reset_index(drop=True)

    colors = sns.color_palette("pastel", len(df))

    # Plot
    plt.figure(figsize=(7, 5))
    x = np.arange(len(df))
    bars = plt.bar(x, df["MI"], color=colors, edgecolor="black")

    plt.errorbar(x, df["MI"], yerr=df["Error"], fmt='none', ecolor='black', capsize=4)

    # Proper x-axis ticks
    plt.xticks(ticks=x, labels=df["Algorithm"], rotation=0)
    plt.ylabel("Mutual Information")
    plt.tight_layout()
    plt.savefig("plot_A_mi_bars.png", dpi=300)
    plt.close()



def plot_combined_bar(boot_dict):
    # Prepare means and stds
    means = {alg: vals.mean() for alg, vals in boot_dict.items()}
    stds  = {alg: vals.std(ddof=1) for alg, vals in boot_dict.items()}

    # Create DataFrame
    df = pd.DataFrame({
        'Algorithm': list(means.keys()),
        'comb': list(means.values()),
        'Error': list(stds.values())
    })

    # Sort for consistent plotting
    df = df.sort_values("comb", ascending=False).reset_index(drop=True)

    # Generate unique colors
    colors = sns.color_palette("pastel", len(df))

    # Plot
    plt.figure(figsize=(7, 5))
    x = np.arange(len(df))
    bars = plt.bar(x, df["comb"], color=colors, edgecolor="black")

    # Add error bars
    plt.errorbar(x, df["comb"], yerr=df["Error"], fmt='none', ecolor='black', capsize=4)

    # Proper x-axis ticks
    plt.xticks(ticks=x, labels=df["Algorithm"], rotation=0)
    plt.ylabel("Weighted Score")
    plt.tight_layout()
    plt.savefig("plot_combined.png", dpi=300)
    plt.close()


def detect_anomalies_kmeans(X_matched, best_combo):
    """
    Detect anomalies using statistical threshold (distance > mean + 3×std) from KMeans centroids.
    Returns a boolean Series.
    """
    kmeans_best = best_combo.get('kmeans')
    dim, k, _ = kmeans_best
    X_reduced = reduce_dims(X_matched, method='umap', n_components=dim)

    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_reduced)

    distances = np.linalg.norm(X_reduced - kmeans.cluster_centers_[labels], axis=1)

    is_anomaly = np.zeros(len(X_matched), dtype=bool)
    for cluster_id in np.unique(labels):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_distances = distances[cluster_indices]
        mean_dist = np.mean(cluster_distances)
        std_dist = np.std(cluster_distances)

        threshold = mean_dist + 3 * std_dist
        outliers = cluster_indices[cluster_distances > threshold]
        is_anomaly[outliers] = True

    is_anomaly_series = pd.Series(is_anomaly, index=X_matched.index)
    return is_anomaly_series, X_reduced, labels



def save_kmeans_anomalies_to_csv(is_anomaly, labels, sample_ids):
    filename = "kmeans_anomalies.csv"

    df = pd.DataFrame({
        "Sample_ID": sample_ids,
        "Cluster_Label": labels,
        "Is_Anomaly": is_anomaly
    })

    df.to_csv(filename, index=False)
    print(f"✅ Saved KMeans anomalies to '{filename}'")


def plot_kmeans_anomalies(X_reduced, labels, is_anomaly):

    plt.figure(figsize=(10, 7))
    n_clusters = len(set(labels))
    palette = get_glasbey_palette(n_clusters)

    # Step 1: 2D embedding
    X_tsne = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_reduced)

    # Step 2: Plot normal samples
    normal_idx = ~is_anomaly
    sns.scatterplot(
        x=X_tsne[normal_idx, 0], y=X_tsne[normal_idx, 1],
        hue=np.array(labels)[normal_idx], palette=palette, s=25, legend=False
    )

    # Step 3: Plot anomalies with different shape
    for i in np.where(is_anomaly)[0]:
        cluster_id = labels[i]
        color = palette[cluster_id]
        plt.scatter(X_tsne[i, 0], X_tsne[i, 1],
                    edgecolor="black", facecolor=color,
                    s=35, marker="X", linewidth=0.5)

    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.savefig("kmeans_anomalies.png", dpi=300)
    plt.close()
    print("✅ Saved KMeans anomaly visualization as 'kmeans_anomalies.png'")



def detect_gmm_anomalies(X_matched, best_combo):

    contamination = 0.02
    gmm_best = best_combo.get('gmm')
    dim, k, _ = gmm_best
    X_reduced = reduce_dims(X_matched, method='umap', n_components=dim)
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(X_reduced)
    gmm_labels = gmm.predict(X_reduced)

    log_probs = gmm.score_samples(X_reduced)  # log-likelihoods
    threshold = np.percentile(log_probs, 100 * contamination)

    anomalies = log_probs < threshold
    return anomalies, gmm_labels, X_reduced


def save_gmm_anomalies_to_csv(is_anomaly, labels, sample_ids):
    filename = "gmm_anomalies.csv"

    df = pd.DataFrame({
        "Sample_ID": sample_ids,
        "Cluster_Label": labels,
        "Is_Anomaly": is_anomaly
    })

    df.to_csv(filename, index=False)
    print(f"✅ Saved GMM anomalies to '{filename}'")



def plot_gmm_anomalies(X_reduced, labels, is_anomaly):
    plt.figure(figsize=(10, 7))

    # Reduce to 2D
    X_tsne = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_reduced)

    # Ensure correct color mapping even with non-contiguous labels
    unique_labels = sorted(set(labels))
    n_clusters = len(set(labels))
    palette = get_glasbey_palette(n_clusters)
    label_to_color = {label: palette[i] for i, label in enumerate(unique_labels)}
    colors = [label_to_color[lbl] for lbl in labels]

    # Plot normal points with consistent coloring
    plt.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], c=colors, s=25, alpha=0.6)

    # Overlay anomalies
    for i, is_out in enumerate(is_anomaly):
        if is_out:
            label = labels[i]
            color = label_to_color.get(label, "black")
            plt.scatter(X_tsne[i, 0], X_tsne[i, 1],
                        edgecolor="black", facecolor=color,
                        s=30, marker="s", linewidth=0.5)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.savefig("gmm_anomalies.png", dpi=300)
    plt.close()
    print("✅ Saved GMM anomaly visualization as 'gmm_anomalies.png'")


def detect_hdbscan_anomalies(X_matched, best_combo):
    contamination = 0.02
    dim, _, _ = best_combo.get("hdbscan", (10, None, None))  # fallback dim
    X_reduced = reduce_dims(X_matched, method='umap', n_components=dim)

    clusterer = HDBSCAN(min_cluster_size=20, min_samples=10, prediction_data=True)
    cluster_labels = clusterer.fit_predict(X_reduced)
    outlier_scores = clusterer.outlier_scores_

    if outlier_scores is None:
        raise ValueError("Outlier scores not available. Check HDBSCAN fit.")

    # Define threshold based on contamination
    threshold = np.percentile(outlier_scores, 100 * (1 - contamination))
    is_anomaly = outlier_scores > threshold

    return is_anomaly, cluster_labels, X_reduced


def plot_hdbscan_anomalies(X_reduced, labels, is_anomaly):
    plt.figure(figsize=(10, 7))
    X_tsne = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_reduced)

    unique_labels = sorted(set(labels))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    n_clusters = len(set(labels))
    index_to_color = get_glasbey_palette(n_clusters)
    label_colors = {label: index_to_color[idx] for label, idx in label_to_index.items()}



    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, palette=label_colors, s=25, legend=False)

    for i, is_out in enumerate(is_anomaly):
        if is_out:
            cluster_label = labels[i]
            c = label_colors[cluster_label]
            plt.scatter(X_tsne[i, 0], X_tsne[i, 1],
                        edgecolor="black", facecolor=c,
                        s=30, marker="s", linewidth=0.5)

    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.savefig("hdbscan_anomalies.png", dpi=300)
    plt.close()
    print("✅ Saved HDBSCAN anomaly visualization as 'hdbscan_anomalies.png'")


def save_hdbscan_anomalies_to_csv(is_anomaly, labels, sample_ids):
    filename = "hdbscan_anomalies.csv"
    import pandas as pd

    df = pd.DataFrame({
        "Sample_ID": sample_ids,
        "Cluster_Label": labels,
        "Is_Anomaly": is_anomaly
    })

    df.to_csv(filename, index=False)
    print(f"✅ Saved HDBSCAN anomalies to '{filename}'")


def analyze_cluster_densities(X_matched, y_labels, code_to_name, best_combo,eps_dim, cluster_algs):
    results = {}
    X_h = X_matched.copy()
    for alg in cluster_algs:
        if alg not in best_combo:
            continue
        dim, k, _ = best_combo[alg]

        X_red = reduce_dims(X_matched, method='umap', n_components=dim)
        if alg == 'kmeans':
            labels = KMeans(n_clusters=k, random_state=42).fit_predict(X_red)
        elif alg == 'gmm' or alg == 'gmm_bic':
            labels = GaussianMixture(n_components=k, random_state=42).fit(X_red).predict(X_red)
        elif alg == 'hdbscan':
            labels = HDBSCAN(min_cluster_size=20, min_samples=10).fit_predict(X_red)
        elif alg == 'dbscan':
            eps = eps_dim[dim]
            labels = DBSCAN(eps=eps, min_samples=5).fit_predict(X_red)
        elif alg == 'hierarchical':
            labels = AgglomerativeClustering(n_clusters=k).fit_predict(X_h)
        else:
            continue

        cluster_density = defaultdict(lambda: defaultdict(int))

        # Count cancer types per cluster
        for lbl, cancer_code in zip(labels, y_labels):
            if lbl == -1:
                continue
            cancer_name = code_to_name[cancer_code]
            cluster_density[lbl][cancer_name] += 1

        # Normalize
        cluster_density_norm = {}
        for clust, counts in cluster_density.items():
            total = sum(counts.values())
            cluster_density_norm[clust] = {ctype: count / total for ctype, count in counts.items()}

        results[alg] = cluster_density_norm

    return results


def summarize_cluster_densities(density_results):
    for alg, clusters in density_results.items():
        print(f"\n📊 {alg.upper()} — Cluster Cancer Type Summary")
        for cluster_id, type_dist in clusters.items():
            sorted_types = sorted(type_dist.items(), key=lambda x: x[1], reverse=True)
            top_cancer = sorted_types[0][0]
            confidence = sorted_types[0][1]
            print(f"  Cluster {cluster_id:>2}: {top_cancer:<25}  ({confidence:.1%} of samples)")



def save_cluster_density_to_csv(density_results, code_to_name):
    rows = []
    filename = "cluster_densities.csv"
    for alg, cluster_map in density_results.items():
        for cluster_id, cancer_dist in cluster_map.items():
            for cancer_code, proportion in cancer_dist.items():
                cancer_name = code_to_name.get(cancer_code, cancer_code)
                rows.append({
                    "Algorithm": alg,
                    "Cluster": cluster_id,
                    "Cancer Type": cancer_name,
                    "Proportion": round(proportion * 100, 4)
                })

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"✅ Cluster densities saved to {filename}")


def get_top_cancer_per_cluster_table(density_results):
    filename = "top_type_table.csv"
    summary = []

    for alg, clusters in density_results.items():
        for clust_id, cancer_dist in clusters.items():
            if not cancer_dist:
                continue
            top_cancer = max(cancer_dist.items(), key=lambda x: x[1])
            summary.append({
                "Algorithm": alg,
                "Cluster": clust_id,
                "Top Cancer Type": top_cancer[0],
                "Proportion": round(top_cancer[1], 3)
            })
    df = pd.DataFrame(summary)
    df.to_csv(filename, index=False)
    print(f"✅ Cluster densities saved to {filename}")
    return df


def identify_misdiagnosed_samples(top_type_df, cluster_labels, true_labels, code_to_name):

    misdiagnosed_dict = {}

    # (Algorithm, cluster) → top cancer type
    cluster_top_map = {
        (row["Algorithm"], row["Cluster"]): row["Top Cancer Type"]
        for _, row in top_type_df.iterrows()
        if row["Proportion"] >= 0.9  # consider only dominant clusters
    }

    for alg, label_series in cluster_labels.items():
        mis_ids, mis_true, mis_cluster, mis_dominant = [], [], [], []

        for sample_id, clust in label_series.items():
            key = (alg, clust)
            if key in cluster_top_map:
                code = true_labels.loc[sample_id]
                true_type = code_to_name[code]
                top_type = cluster_top_map[key]
                if true_type != top_type:
                    mis_ids.append(sample_id)
                    mis_true.append(true_type)
                    mis_cluster.append(clust)
                    mis_dominant.append(top_type)

        df = pd.DataFrame({
            "Sample_ID": mis_ids,
            "Cluster": mis_cluster,
            "TrueType": mis_true,
            "DominantClusterType": mis_dominant,
            "Algorithm": alg
        })
        misdiagnosed_dict[alg] = df

    return misdiagnosed_dict


def analyze_anomaly_misdiagnosis_relationship(anomaly_files, misdiagnosed_file):
    output_file = 'fisher_results.csv'
    results = []
    misdiagnosed_df = pd.read_csv(misdiagnosed_file)

    for file in anomaly_files:
        df = pd.read_csv(file)
        alg = file.split("_")[0]  # assuming 'kmeans_bool_anomalies.csv' → 'kmeans'

        # Add column for whether the sample is misdiagnosed
        df['Is_Misdiagnosed'] = df['Sample_ID'].isin(
            misdiagnosed_df[misdiagnosed_df["Algorithm"] == alg]["Sample_ID"]
        )

        # Construct contingency table
        a = len(df[(df["Is_Anomaly"] == True) & (df["Is_Misdiagnosed"] == True)])
        b = len(df[(df["Is_Anomaly"] == True) & (df["Is_Misdiagnosed"] == False)])
        c = len(df[(df["Is_Anomaly"] == False) & (df["Is_Misdiagnosed"] == True)])
        d = len(df[(df["Is_Anomaly"] == False) & (df["Is_Misdiagnosed"] == False)])

        contingency = [[a, b], [c, d]]

        # Fisher exact test
        odds_ratio, p_value = fisher_exact(contingency)

        results.append({
            "Algorithm": alg,
            "Anomaly_and_Misdiagnosed": a,
            "Anomaly_only": b,
            "Misdiagnosed_only": c,
            "Normal": d,
            "Odds_Ratio": round(odds_ratio, 3),
            "P_Value": f"{p_value:.2e}"
        })

        print(f"📊 {alg}: Odds Ratio = {odds_ratio:.3f}, P = {p_value:.2e}")

    # Save to CSV
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False)
    print(f"\n✅ Fisher results saved to '{output_file}'")




def save_misdiagnosed_to_csv(misdiagnosed_dict):
    filename = "misdiagnosed_samples.csv"
    all_df = pd.concat(misdiagnosed_dict.values(), ignore_index=True)

    all_df.to_csv(filename, index=False)
    print(f"✅ Saved misdiagnosed samples to: {filename}")


def count_dominant_clusters(top_type_df, threshold=0.9):
    counts = (
        top_type_df[top_type_df["Proportion"] >= threshold]
        .groupby("Algorithm")
        .size()
        .reset_index(name="DominantClusterCount")
    )

    print("\n🔍 Dominant clusters per algorithm:")
    for _, row in counts.iterrows():
        print(f"{row['Algorithm']}: {row['DominantClusterCount']} clusters with ≥{int(threshold * 100)}% dominance")

    return counts



def main():

    #CLUSTERING AND DIMENSIONS

    X_scaled = prepare_tcga_pancan()
    feature_cols = X_scaled.columns
    label_col = '_primary_disease'
    X_merged = merge_labels(X_scaled)
    X_matched = X_merged[feature_cols]
    X_matched.to_csv("X_scaled")
    y_labels_named = X_merged[label_col]
    y_labels_named.to_csv("y_labels.csv")
    print("SAVED")
    y_labels_cat = X_merged[label_col].astype('category')
    y_labels = y_labels_cat.cat.codes
    code_to_name = dict(enumerate(y_labels_cat.cat.categories))

    print(f"Features: {X_matched.shape}, Labels: {y_labels.shape}")

    print("Unique labels:", y_labels.unique())
    print("Number of unique labels:", y_labels.nunique())

    X_train, X_test, y_train, y_test = train_test_split(X_matched, y_labels, test_size=0.4, random_state=42, stratify=y_labels)

    best_dim_km, best_k_km = pick_kmeans_dim_k(X_train, dims=DIMS_REDUCE, ks=KS)

    best_k_gmm = find_best_k_gmm(X_scaled, KS)
    print(f"Best k and dimension for new method are k = {best_k_km}, dim = {best_dim_km}")
    print(f"Best k in gmm BIC: {best_k_gmm}")

    print(f"\n🔍 Clustering with UMAP...")

    results, boot_sil, boot_mi, boot_combined, best_combo, eps_dim, all_cluster_labels = run_all(X_matched, X_train, X_test, best_k_km, best_dim_km,best_k_gmm, y_train, y_test, n_boot=50, seed=42)

    pd.DataFrame(list(eps_dim.items()), columns=["Dim", "Eps"]).to_csv("eps_dim.csv", index=False)

    best_combo_df = pd.DataFrame(
        [(alg, dim, k, mode) for alg, (dim, k, mode) in best_combo.items()],
        columns=["Algorithm", "Dim", "K", "Mode"]
    )

    # Save to CSV
    best_combo_df.to_csv("best_combo.csv", index=False)
    print("✅ Saved best_combo to 'best_combo.csv'")

    plot_silhouette_bar(boot_sil)
    plot_mi_bar(boot_mi)
    plot_combined_bar(boot_combined)

    plot_heatmaps(results)
    print_best_params(results)
    plot_dendrogram(X_matched)
    visualize_clusters_tsne(X_matched, results, eps_dim, method='umap')


    #BOOTSTRAPS


    print_bootstrap_summary(boot_sil ,'sil')
    print_bootstrap_summary(boot_mi,'mi')
    print_bootstrap_summary(boot_combined,'comb')

    save_best_params_to_csv(results)
    save_bootstrap_summary_to_csv(boot_sil, 'sil')
    save_bootstrap_summary_to_csv(boot_mi, 'mi')
    save_bootstrap_summary_to_csv(boot_combined, 'comb')

    save_raw_bootstrap_to_csv(boot_sil, 'sil')
    save_raw_bootstrap_to_csv(boot_mi, 'mi')
    save_raw_bootstrap_to_csv(boot_combined, 'comb')


    #DENSITY
    print("Performing density analysis")
    density_results = analyze_cluster_densities(X_matched, y_labels, code_to_name, best_combo, eps_dim, CLUSTER_ALGS)
    summarize_cluster_densities(density_results)
    save_cluster_density_to_csv(density_results, code_to_name)
    top_type_df = get_top_cancer_per_cluster_table(density_results)



    mis_dic = identify_misdiagnosed_samples(top_type_df, all_cluster_labels, y_labels, code_to_name)
    save_misdiagnosed_to_csv(mis_dic)
    dominant_counts =count_dominant_clusters(top_type_df, threshold=0.9)
    print(dominant_counts)
    dominant_counts.to_csv("dominant_cluster_counts.csv", index=False)

    #ANOMALY DETECTION - KMEANS
    print("Running K-Means anomaly detection")
    is_anomaly, X_reduced_km, labels = detect_anomalies_kmeans(X_matched, best_combo)
    plot_kmeans_anomalies(X_reduced_km, labels, is_anomaly)
    save_kmeans_anomalies_to_csv(is_anomaly, labels, X_matched.index)


    #ANOMALY DETECTION - GMM
    print("Running GMM anomaly detection")
    anomalies_gmm, gmm_labels, X_reduced_gmm = detect_gmm_anomalies(X_matched, best_combo)
    plot_gmm_anomalies(X_reduced_gmm, gmm_labels, anomalies_gmm)
    save_gmm_anomalies_to_csv(anomalies_gmm, labels, X_matched.index)

    #ANOMALY DETECTION - HDBSCAN
    print("Running HDBSCAN anomaly detection")
    anomalies_hdbscan, hdbscan_labels, X_reduced_hdbscan = detect_hdbscan_anomalies(X_matched, best_combo)
    plot_hdbscan_anomalies(X_reduced_hdbscan, hdbscan_labels, anomalies_hdbscan)
    save_hdbscan_anomalies_to_csv(anomalies_hdbscan, hdbscan_labels, X_matched.index)





main()




def analyze_hdbscan_noise_points():
    # Load cluster labels and true types
    cluster_file = "all_clusters_labels.csv"
    labels_file = "y_labels.csv"
    output_file = "hdbscan_noise_type_distribution.csv"
    clusters_df = pd.read_csv(cluster_file, index_col=0)
    y_labels = pd.read_csv(labels_file, index_col=0).squeeze()

    # Clean up sample IDs
    clusters_df.index = clusters_df.index.str.replace(r"\.1$", "", regex=True)
    y_labels.index = y_labels.index.str.replace(r"\.1$", "", regex=True)

    # Ensure alignment
    common_idx = clusters_df.index.intersection(y_labels.index)
    noise_labels = clusters_df.loc[common_idx, "hdbscan"]
    noise_mask = noise_labels == -1

    if noise_mask.sum() == 0:
        print("⚠️ No noise points found in HDBSCAN clustering.")
        return

    # Subset to noise points and get their cancer types
    noise_samples = y_labels.loc[noise_mask]

    # Count and save
    counts = noise_samples.value_counts()
    counts_df = counts.reset_index()
    counts_df.columns = ["CancerType", "Count"]
    counts_df.to_csv(output_file, index=False)

    # Print summary
    most_common = counts.idxmax()
    proportion = counts.max() / counts.sum()
    print("📊 HDBSCAN Noise Point Cancer Type Distribution saved to", output_file)
    print(f"🔍 Most prevalent type: {most_common} ({proportion:.2%} of noise samples)")

    return counts_df


analyze_hdbscan_noise_points()

def fisher():
    anomalies_files = [
          "kmeans_anomalies.csv",
          "gmm_anomalies.csv",
         "hdbscan_anomalies.csv"
        ]

    misdiagnosed_df_file = "misdiagnosed_samples.csv"
    analyze_anomaly_misdiagnosis_relationship(anomalies_files, misdiagnosed_df_file)

fisher()


def analyze_most_misdiagnosed_by_algorithm():
    # Load the misdiagnosed data
    df = pd.read_csv("misdiagnosed_samples.csv")

    if "Algorithm" not in df.columns or "TrueType" not in df.columns:
        raise ValueError("CSV must contain 'Algorithm' and 'TrueType' columns.")

    # Group by Algorithm and TrueType, and count occurrences
    grouped = (
        df.groupby(["Algorithm", "TrueType"])
          .size()
          .reset_index(name="MisdiagnosedCount")
          .sort_values(["Algorithm", "MisdiagnosedCount"], ascending=[True, False])
    )

    # Save to CSV
    grouped.to_csv("most_misdiagnosed_by_algorithm.csv", index=False)
    print("✅ Saved most misdiagnosed types by algorithm to 'most_misdiagnosed_by_algorithm.csv'.")

    return grouped


analyze_most_misdiagnosed_by_algorithm()


def investigate_misdiagnosed_type(misdiagnosed_df, cluster_labels_dict, top_type_df):
    cancer_type = 'lung squamous cell carcinoma'
    alg = 'hdbscan'

    # Filter misdiagnosed samples of the given cancer type
    sub_df = misdiagnosed_df[(misdiagnosed_df['TrueType'] == cancer_type) & (misdiagnosed_df['Algorithm'] == alg)]
    cluster_labels = cluster_labels_dict[alg]

    # Match sample → cluster
    clusters = cluster_labels.loc[sub_df['Sample_ID']]
    sub_df = sub_df.copy()
    sub_df['Cluster'] = clusters.values

    # Count how many misdiagnosed samples fall into each cluster
    cluster_counts = sub_df['Cluster'].value_counts().sort_values(ascending=False)

    # Map each cluster to its dominant type (if any)
    cluster_to_top = top_type_df[top_type_df['Algorithm'] == alg].set_index('Cluster')['Top Cancer Type'].to_dict()

    # Summarize
    results = []
    for cluster, count in cluster_counts.items():
        dominant = cluster_to_top.get(cluster, 'N/A')
        results.append({
            "Cluster": cluster,
            "Misdiagnosed Count": count,
            "Dominant Type": dominant
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(f"{alg}_{cancer_type}_misplacement_summary.csv", index=False)
    print(f"✅ Saved misdiagnosis summary for {cancer_type} ({alg})")

    return result_df


df = pd.read_csv("misdiagnosed_samples.csv")
cluster_df = pd.read_csv("all_clusters_labels.csv", index_col=0)
cluster_df.reset_index(inplace=True)
cluster_df.rename(columns={"index": "Sample_ID"}, inplace=True)
cluster_df.set_index("Sample_ID", inplace=True)

print(cluster_df.columns)

tpdf = pd.read_csv("top_type_table.csv")
investigate_misdiagnosed_type(df, cluster_df, tpdf)


def detect_potential_subtypes(df, purity_thresh=0.90):
    output = "potential_subtypes.csv"

    # Filter only highly pure clusters
    high_purity = df[df["Proportion"] >= purity_thresh].copy()

    # Group by Algorithm and Top Cancer Type
    grouped = (
        high_purity.groupby(["Algorithm", "Top Cancer Type"])
        .agg(
            NumClusters=("Cluster", "nunique"),
            ClusterIDs=("Cluster", lambda x: list(x)),
            Proportions=("Proportion", lambda x: [round(val, 3) for val in x])
        )
        .reset_index()
    )

    # Keep only cancer types that appear in ≥2 clusters
    candidate_subtypes = grouped[grouped["NumClusters"] >= 2]

    # Save
    candidate_subtypes.to_csv(output, index=False)
    print(f"✅ Saved subtype candidates to {output}")
    return candidate_subtypes


subtype_df = detect_potential_subtypes(tpdf, 0.90)



def visualize_subtype_candidates():
    X = pd.read_csv("X_scaled.csv", index_col=0)
    best_combo_df = pd.read_csv("best_combo.csv")
    eps_df = pd.read_csv("eps_dim.csv")
    subtype_df = pd.read_csv("potential_subtypes.csv")

    # Prepare best_combo dict
    best_combo = {
        row["Algorithm"]: (
            int(row["Dim"]),
            None if pd.isna(row["K"]) or row["K"] == 'auto' else int(row["K"]),
            row["Mode"]
        )
        for _, row in best_combo_df.iterrows()
    }
    eps_dim = dict(zip(eps_df["Dim"], eps_df["Eps"]))

    os.makedirs("subtype_plots_umap", exist_ok=True)

    for cancer_type in subtype_df["Top Cancer Type"].unique():
        subset = subtype_df[subtype_df["Top Cancer Type"] == cancer_type]
        if subset.empty:
            continue

        for alg in subset["Algorithm"].unique():
            if alg not in best_combo:
                continue
            dim, k, mode = best_combo[alg]
            X_red = reduce_dims(X, method='umap', n_components=dim)

            # Clustering
            if alg == "kmeans":
                labels = KMeans(n_clusters=k, random_state=42).fit_predict(X_red)
            elif alg in ("gmm", "gmm_bic"):
                labels = GaussianMixture(n_components=k, random_state=42).fit(X_red).predict(X_red)
            elif alg == "dbscan":
                eps = eps_dim.get(dim, 5)
                labels = DBSCAN(eps=eps, min_samples=5).fit_predict(X_red)
            elif alg == "hdbscan":
                labels = HDBSCAN(min_cluster_size=20, min_samples=10).fit_predict(X_red)
            else:
                continue

            # Normalize labels for color mapping
            encoder = LabelEncoder()
            norm_labels = encoder.fit_transform(labels)
            palette = get_glasbey_palette(len(np.unique(norm_labels)))

            X_2d = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_red)

            # Plot all points colored by cluster
            plt.figure(figsize=(10, 7))
            sns.scatterplot(
                x=X_2d[:, 0], y=X_2d[:, 1], hue=norm_labels,
                palette=palette, s=25, legend=False, linewidth=0
            )

            # Highlight subtype clusters
            subtype_clusters = subset[subset["Algorithm"] == alg]["ClusterIDs"]
            clusters_to_highlight = set()
            for clist in subtype_clusters:
                if isinstance(clist, str):
                    clusters_to_highlight.update(eval(clist))
                elif isinstance(clist, list):
                    clusters_to_highlight.update(clist)

            for cluster_id in clusters_to_highlight:
                indices = np.where(labels == cluster_id)[0]
                for i in indices:
                    plt.scatter(
                        X_2d[i, 0], X_2d[i, 1],
                        facecolor=palette[norm_labels[i]], edgecolor="black",
                        marker="s", s=60, linewidth=0.7
                    )

            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.tight_layout()
            fname = f"subtype_plots_umap/{alg}_{cancer_type.replace(' ', '_')}_highlight.png"
            plt.savefig(fname, dpi=300)
            plt.close()
            print(f"✅ Saved: {fname}")









def visualize_clusters_vs_true_types(highlight_types, num):


    # Load data
    X = pd.read_csv("X_scaled", index_col=0)
    y_labels = pd.read_csv("y_labels.csv", index_col=0).squeeze()
    clusters_df = pd.read_csv("all_clusters_labels.csv", index_col=0)
    best_combo_df = pd.read_csv("best_combo.csv")
    eps_df = pd.read_csv("eps_dim.csv")
    eps_dim = dict(zip(eps_df["Dim"], eps_df["Eps"]))

    for df in [X, y_labels, clusters_df]:
        df.index = df.index.str.replace(r"\.1$", "", regex=True)

    dir_name = ""
    if num == 1:
        dir_name = "cho"
    if num == 2:
        dir_name = "breast_invasive_carcinoma"
    if num == 3:
        dir_name = "hn"
    if num == 4:
        dir_name = "test_legend"
    if num == 5:
        dir_name = "lung_adenocarcinoma"
    os.makedirs(dir_name, exist_ok=True)

    if highlight_types and isinstance(highlight_types, str):
        highlight_types = [highlight_types]

    for _, row in best_combo_df.iterrows():
        alg = row["Algorithm"]
        dim = int(row["Dim"])
        k = None if pd.isna(row["K"]) else int(row["K"])
        if alg not in clusters_df.columns:
            continue

        X_red = reduce_dims(X, method='umap', n_components=dim)
        X_tsne = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_red)

        if alg == "kmeans":
            labels = KMeans(n_clusters=k, random_state=42).fit_predict(X_red)
        elif alg in ("gmm", "gmm_bic"):
            labels = GaussianMixture(n_components=k, random_state=42).fit(X_red).predict(X_red)
        elif alg == "dbscan":
            labels = DBSCAN(eps=eps_dim[dim], min_samples=5).fit_predict(X_red)
        elif alg == "hdbscan":
            labels = HDBSCAN(min_cluster_size=20, min_samples=10).fit_predict(X_red)
        else:
            continue

        common_idx = X.index.intersection(y_labels.index)
        df_vis = pd.DataFrame({
            "TSNE1": X_tsne[:, 0],
            "TSNE2": X_tsne[:, 1],
            "Cluster": labels,
            "CancerType": y_labels.loc[common_idx].values
        }, index=common_idx)

        if highlight_types:
            df_vis["Marker"] = df_vis["CancerType"].apply(
                lambda ct: "highlight" if ct in highlight_types else "normal"
            )
        else:
            df_vis["Marker"] = "normal"

        cluster_palette = get_glasbey_palette(len(set(labels)) - (1 if -1 in labels else 0))
        cancer_palette = get_glasbey_palette(df_vis["CancerType"].nunique())

        plt.figure(figsize=(10, 6))

        # Background clusters
        sns.scatterplot(
            x="TSNE1", y="TSNE2",
            hue="Cluster", data=df_vis,
            palette=cluster_palette,
            s=150, alpha=0.25,
            edgecolor=None, linewidth=0,
            legend=False, zorder=1
        )

        # Foreground cancer types with styles
        markers = {"highlight": "X", "normal": "o"}
        sizes = {"highlight": 50, "normal": 12}
        scatter = sns.scatterplot(
            x="TSNE1", y="TSNE2",
            hue="CancerType",
            style="Marker",
            data=df_vis,
            palette=cancer_palette,
            markers=markers,
            size="Marker",
            sizes=sizes,
            edgecolor="black",
            linewidth=0.2,
            alpha=0.9,
            legend=False,
            zorder=2
        )

        # Legend
        handles, labels_ = scatter.get_legend_handles_labels()
        legend_fig = plt.figure(figsize=(8, 6))
        legend_ax = legend_fig.add_subplot(111)
        legend_ax.legend(handles, labels_, loc='center', frameon=False)
        legend_ax.axis('off')
        hname = "_".join(highlight_types) if highlight_types else "none"
        legend_path = os.path.join(dir_name, f"{alg}_highlight_{hname}_legend.png")
        legend_fig.savefig(legend_path, dpi=300, bbox_inches='tight')
        plt.close(legend_fig)

        #plt.title(f"{alg.upper()} – Cluster vs. Cancer Type"
         #         f"{' (Highlight: ' + ', '.join(highlight_types) + ')' if highlight_types else ''}")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.tight_layout()
        fname = os.path.join(dir_name, f"{alg}_highlight_{hname}.png")
        plt.savefig(fname, dpi=300)
        plt.close()
        print(f"✅ Saved: {fname} and legend.")


visualize_clusters_vs_true_types(highlight_types=["cholangiocarcinoma"], num=1)
visualize_clusters_vs_true_types(highlight_types=["breast invasive carcinoma"], num=2)
visualize_clusters_vs_true_types(highlight_types=["head & neck squamous cell carcinoma"], num=3)




def check_normality(grouped_scores, alpha=0.05):
    print("\n🔍 Normality Test Results (Shapiro–Wilk):")
    normality_results = {}
    for alg, scores in grouped_scores.items():
        if len(scores) < 3:
            print(f"{alg}: Not enough samples")
            continue
        stat, p_val = shapiro(scores)
        normal = p_val > alpha
        normality_results[alg] = normal
        print(f"{alg:<10} p = {p_val:.4f} → {'✅ Normal' if normal else '❌ Not Normal'}")
    return normality_results


def plot_qq(grouped_scores):
    for alg, scores in grouped_scores.items():
        plt.figure()
        probplot(scores, dist="norm", plot=plt)
        plt.tight_layout()
        plt.savefig(f"qqplot_{alg}.png")
        plt.close()


def run_statistical_tests(bootstrap_csv_path, top_n):
    alpha = 0.05
    output_csv = "statistical_results.csv"
    df = pd.read_csv(bootstrap_csv_path)
    metric_column = "score"
    if metric_column not in df.columns or 'algorithm' not in df.columns:
        raise ValueError("CSV must contain 'algorithm' and the specified metric column")

    grouped = df.groupby("algorithm")[metric_column].apply(list).to_dict()
    mean_scores = {alg: sum(vals)/len(vals) for alg, vals in grouped.items()}
    summary_df = pd.DataFrame([{"algorithm": k, "mean_score": v} for k, v in mean_scores.items()])
    summary_df.sort_values(by="mean_score", ascending=False, inplace=True)
    summary_df.to_csv("algorithm_mean_scores.csv", index=False)

    # Check normality
    normality_flags = check_normality(grouped, alpha)
    plot_qq(grouped)

    # Kruskal-Wallis test
    kw_stat, kw_p = kruskal(*grouped.values())
    print("\n🔬 Kruskal-Wallis H-test")
    print(f"H = {kw_stat:.4f}, p = {kw_p:.4e}")

    # Prepare for writing
    sorted_algos = summary_df["algorithm"].tolist()[:top_n]
    p_values = {}
    pairwise_results = []

    with open("test_summary.txt", "w") as f:
        f.write(f"Kruskal-Wallis H = {kw_stat:.4f}, p = {kw_p:.4e}\n")
        print("\n📊 Pairwise Comparisons:")
        for i in range(len(sorted_algos)):
            for j in range(i + 1, len(sorted_algos)):
                a1, a2 = sorted_algos[i], sorted_algos[j]
                x, y = grouped[a1], grouped[a2]

                try:
                    if normality_flags.get(a1, False) and normality_flags.get(a2, False):
                        stat, p_val = ttest_rel(x, y)
                        test_name = "Paired t-test"
                    else:
                        diff = np.array(x) - np.array(y)
                        if np.allclose(diff, 0):
                            p_val = 1.0
                            test_name = "Wilcoxon (all zero diff)"
                        else:
                            stat, p_val = wilcoxon(x,y)
                            test_name = "Wilcoxon signed-rank"
                except:
                    stat, p_val = mannwhitneyu(x, y, alternative='greater')
                    test_name = "Mann-Whitney U"

                p_values[(a1, a2)] = p_val
                result_text = f"{a1} > {a2} [{test_name}]: p = {p_val:.4e}"
                print(result_text)
                f.write(result_text + "\n")
                pairwise_results.append({
                    "algorithm_1": a1,
                    "algorithm_2": a2,
                    "p_value": p_val,
                    "test": test_name
                })

    # Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=[grouped[alg] for alg in summary_df["algorithm"]], notch=True)
    plt.xticks(ticks=range(len(grouped)), labels=summary_df["algorithm"], rotation=45)
    plt.ylabel(metric_column)
    plt.tight_layout()
    plt.savefig("bootstrap_boxplot.png", dpi=300)
    plt.close()

    result_df = pd.DataFrame(pairwise_results)
    result_df.to_csv(output_csv, index=False)
    print(f"\n✅ Statistical test results saved to {output_csv}")

    return summary_df, p_values


summary_df, p_values = run_statistical_tests("combined_bootstrap_full.csv", top_n=7)


