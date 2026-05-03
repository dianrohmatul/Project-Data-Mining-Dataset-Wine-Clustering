"""
app.py — Wine Clustering Interactive Dashboard
Jalankan: streamlit run app.py
"""
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import matplotlib
matplotlib.use('Agg')
 
from sklearn.datasets import load_wine
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import (KMeans, AgglomerativeClustering,
                             DBSCAN, SpectralClustering)
from sklearn.metrics import silhouette_score, davies_bouldin_score
 
try:
    import skfuzzy as fuzz
    HAS_FUZZY = True
except ImportError:
    HAS_FUZZY = False

# Konfigurasi halaman
st.set_page_config(page_title="Wine Clustering Dashboard", layout="wide")
st.title("Wine Clustering — Interactive Dashboard")
st.markdown("""
- **Nama:** Dian Rohmatul Islam  
- **Kelas:** 2024 A  
- **Semester:** 4  
- **Mata Kuliah:** Data Mining  
- **Program Studi: S1 Sains Data** 
- **Fakultas: Matematika dan Ilmu Pengetahuan Alam**
- **Kampus: Universitas Negeri Surabaya**
""")

# [1] Load + Scale + PCA — semua di-cache agar tidak diulang setiap interaksi
@st.cache_data
def load_and_preprocess():
    """Muat dataset, scale, dan komputasi PCA satu kali."""
    data     = load_wine()
    X        = pd.DataFrame(data.data, columns=data.feature_names)
    y        = pd.Series(data.target, name="target")
 
    scaler   = RobustScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), columns=X.columns
    )
 
    # PCA di-fit sekali di sini, bukan di setiap render
    pca   = PCA(n_components=2, random_state=42)
    X_vis = pca.fit_transform(X_scaled)          # numpy array (178, 2)
    var_explained = pca.explained_variance_ratio_
 
    return X, y, X_scaled, X_vis, var_explained
 
 
X, y, X_scaled, X_vis, var_explained = load_and_preprocess()
 
# [2] Sidebar — kontrol
st.sidebar.title("Pengaturan Clustering")
st.write("Dataset shape:", X.shape)

with st.sidebar:
    st.header("Controls")
 
    method_options = (
        ["KMeans", "Hierarchical", "DBSCAN", "Spectral"]
        + (["Fuzzy C-Means"] if HAS_FUZZY else [])
    )
    method = st.selectbox("Clustering Method", method_options)
 
    n_clusters = None
    eps        = None
    min_samples = None
 
    if method in ["KMeans", "Hierarchical", "Spectral", "Fuzzy C-Means"]:
        n_clusters = st.slider("n_clusters", 2, 6, 3)
 
    if method == "DBSCAN":
        eps         = st.number_input("DBSCAN eps",   min_value=0.1,
                                      max_value=10.0, value=1.6, step=0.1)
        min_samples = st.slider("DBSCAN min_samples", 1, 20, 5)
 
    # Pilih fitur untuk scatter
    feat_cols = list(X.columns)
    feature_x = st.selectbox("X axis", feat_cols,
                              index=feat_cols.index("alcohol")
                              if "alcohol" in feat_cols else 0)
    feature_y = st.selectbox("Y axis", feat_cols,
                              index=feat_cols.index("proline")
                              if "proline" in feat_cols else 1)
 
    show_original = st.checkbox("Tampilkan label asli (warna)", value=False)
    run_button    = st.button("Run Clustering", type="primary")

# [3] Fungsi clustering
def run_clustering(X_sc, method, n_clusters=None, eps=None, min_samples=None):
    np.random.seed(42)
 
    if method == "KMeans":
        model  = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = model.fit_predict(X_sc)
        centers = model.cluster_centers_
        return labels, centers
 
    if method == "Hierarchical":
        labels = AgglomerativeClustering(
            n_clusters=n_clusters).fit_predict(X_sc)
 
    elif method == "DBSCAN":
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_sc)
 
    elif method == "Spectral":
        labels = SpectralClustering(
            n_clusters=n_clusters, affinity="nearest_neighbors",
            random_state=42).fit_predict(X_sc)
 
    elif method == "Fuzzy C-Means" and HAS_FUZZY:
        _, u, *_ = fuzz.cluster.cmeans(
            X_sc.T.values, c=n_clusters, m=2,
            error=0.005, maxiter=1000
        )
        labels = np.argmax(u, axis=0)
 
    else:
        labels = np.zeros(len(X_sc), dtype=int)
 
    return labels, None   # None = tidak ada centroid
 
# [4] Placeholder sebelum clustering dijalankan
if not run_button:
    st.info("Pilih metode dan parameter di sidebar, lalu klik **Run Clustering**.")
    st.markdown(
        f"Dataset: **{len(X)} sampel · {len(X.columns)} fitur** · "
        f"PCA 2D menangkap **{var_explained.sum()*100:.1f}%** variansi total."
    )
    st.stop()   # hentikan render; tidak ada scatter kosong yang membingungkan
 
# [5] Jalankan clustering
with st.spinner(f"Menjalankan {method}..."):
    labels, centers = run_clustering(
        X_scaled, method,
        n_clusters=n_clusters, eps=eps, min_samples=min_samples
    )

# [6] Tampilkan hasil
col1, col2 = st.columns([2, 1])
 
with col1:
    fig, ax = plt.subplots(figsize=(8, 6))
 
    if show_original:
        sc_orig = ax.scatter(X_vis[:, 0], X_vis[:, 1],
                             c=y, cmap="Set1", alpha=0.4, s=25,
                             label="Label asli")
        ax.legend(*sc_orig.legend_elements(), title="Asli", loc="upper left")
 
    sc = ax.scatter(X_vis[:, 0], X_vis[:, 1],
                    c=labels, cmap="tab10",
                    edgecolor="k", linewidth=0.2,
                    alpha=0.85, s=40)
 
    if centers is not None:
        pca_eval = PCA(n_components=2, random_state=42)
        pca_eval.fit(X_scaled)       # re-fit untuk transform centroid
        c2d = pca_eval.transform(centers)
        ax.scatter(c2d[:, 0], c2d[:, 1], marker="X",
                   s=250, c="black", zorder=5, label="Centroid")
        ax.legend(loc="upper right")
 
    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}%)")
    ax.set_title(f"PCA 2D — {method}")
    st.pyplot(fig)
 
with col2:
    st.subheader("Ringkasan")
 
    # Distribusi cluster
    counts = pd.Series(labels).value_counts().sort_index()
    st.write("**Distribusi cluster:**")
    st.dataframe(counts.rename("Jumlah").to_frame(), use_container_width=True)
 
    # Silhouette score — filter noise untuk DBSCAN
    mask = labels != -1
    valid_labels = labels[mask]
    X_valid      = X_scaled.values[mask]
 
    if len(set(valid_labels)) > 1 and len(valid_labels) > 1:
        sil = silhouette_score(X_valid, valid_labels)
        st.metric("Silhouette Score", f"{sil:.3f}",
                  help="Semakin tinggi semakin baik (maks 1.0). "
                       "Untuk DBSCAN, noise (-1) tidak ikut dihitung.")
 
        if method != "DBSCAN":
            dbi = davies_bouldin_score(X_scaled, labels)
            st.metric("Davies-Bouldin Index", f"{dbi:.3f}",
                      help="Semakin kecil semakin baik.")
    else:
        st.warning("Tidak cukup cluster untuk menghitung Silhouette Score. "
                   "Coba ubah parameter.")
 
    if method == "DBSCAN":
        n_noise = (labels == -1).sum()
        st.metric("Jumlah Noise", n_noise,
                  help=f"{n_noise/len(labels)*100:.1f}% dari total data.")
        
    if sil > 0.5:
        st.success("Cluster sangat baik")
    
    elif sil > 0.25:
        st.info("Cluster cukup baik")
    
    else:
        st.warning("Cluster kurang optimal")
 
# [7] Scatter fitur pilihan
st.markdown("---")
st.subheader("Scatter fitur")
fig2, ax2 = plt.subplots(figsize=(8, 4))
sns.scatterplot(
    x=X[feature_x], y=X[feature_y],
    hue=labels, palette="tab10", ax=ax2,
    edgecolor="k", linewidth=0.2
)
ax2.set_xlabel(feature_x)
ax2.set_ylabel(feature_y)
ax2.set_title(f"{feature_x} vs {feature_y} — {method}")
st.pyplot(fig2)

# [8] Profil rata-rata per cluster (heatmap)
st.subheader("Profil rata-rata fitur per cluster")
temp = X_scaled.copy()
temp["Cluster"] = labels
summary = temp.groupby("Cluster").mean().round(2)
 
fig3, ax3 = plt.subplots(figsize=(12, max(3, len(summary) * 0.8)))
sns.heatmap(summary, annot=True, fmt=".2f", cmap="coolwarm",
            ax=ax3, cbar=True)
ax3.set_title(f"Profil Cluster — {method}")
st.pyplot(fig3)
 
# [9] Tabel data + download
st.subheader("Data dengan label cluster")
df_out = X.copy()
df_out["target"]  = y.values
df_out["cluster"] = labels
st.dataframe(df_out, use_container_width=True, height=300)
 
csv = df_out.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download data berlabel (CSV)",
    data=csv,
    file_name=f"wine_clusters_{method}.csv",
    mime="text/csv"
)
 
# Footer
st.markdown("---")
st.caption(
    "Dataset: UCI Wine (id=109) · "
    f"Fuzzy C-Means: {'tersedia' if HAS_FUZZY else 'tidak tersedia (install scikit-fuzzy)'}."
)
 
