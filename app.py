import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score

try:
    import skfuzzy as fuzz
    HAS_FUZZY = True
except Exception:
    HAS_FUZZY = False

st.set_page_config(page_title="Wine Clustering Dashboard", layout="wide")

st.title("Wine Clustering — Interactive Dashboard")


@st.cache_data
def load_data():
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    return X, y

X, y = load_data()

with st.sidebar:
    st.header("Controls")
    method = st.selectbox(
        "Clustering Method",
        ['KMeans', 'Hierarchical', 'DBSCAN', 'Spectral'] + (['Fuzzy C-Means'] if HAS_FUZZY else [])
    )

    n_clusters = st.slider("n_clusters", 2, 6, 3) if method in ['KMeans', 'Hierarchical', 'Spectral', 'Fuzzy C-Means'] else None
    eps = st.number_input("DBSCAN eps", min_value=0.1, max_value=10.0, value=1.6, step=0.1) if method == 'DBSCAN' else None
    min_samples = st.slider("DBSCAN min_samples", 1, 20, 5) if method == 'DBSCAN' else None

    feature_x = st.selectbox("X axis (feature)", X.columns, index=list(X.columns).index('alcohol') if 'alcohol' in X.columns else 0)
    feature_y = st.selectbox("Y axis (feature)", X.columns, index=list(X.columns).index('proline') if 'proline' in X.columns else 1)

    show_original = st.checkbox("Color by original class", value=True)
    run_button = st.button("Run Clustering")


def run_clustering(X_scaled, method, n_clusters=None, eps=None, min_samples=None):
    if method == 'KMeans':
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(X_scaled)
    elif method == 'Hierarchical':
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X_scaled)
    elif method == 'DBSCAN':
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X_scaled)
    elif method == 'Spectral':
        model = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
        labels = model.fit_predict(X_scaled)
    elif method == 'Fuzzy C-Means' and HAS_FUZZY:
        X_arr = X_scaled.T.values
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(X_arr, c=n_clusters, m=2, error=0.005, maxiter=1000)
        labels = np.argmax(u, axis=0)
    else:
        labels = np.zeros(len(X_scaled), dtype=int)

    return labels


with st.spinner('Preprocessing data...'):
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

if run_button:
    labels = run_clustering(X_scaled, method, n_clusters=n_clusters, eps=eps, min_samples=min_samples)
else:
    labels = np.full(len(X), -1)

X_vis = PCA(n_components=2).fit_transform(X_scaled)

col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(8,6))
    if show_original:
        scatter = ax.scatter(X_vis[:,0], X_vis[:,1], c=y, cmap='Set1', alpha=0.8)
        legend1 = ax.legend(*scatter.legend_elements(), title="Original")
        ax.add_artist(legend1)

    if run_button:
        # color by cluster
        scatter2 = ax.scatter(X_vis[:,0], X_vis[:,1], c=labels, cmap='tab10', marker='o', edgecolor='k', linewidth=0.2, alpha=0.9)
        ax.set_title(f'PCA (2D) — {method}')
    else:
        ax.set_title('PCA (2D) — (no clustering run yet)')

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    st.pyplot(fig)

with col2:
    st.subheader('Parameters')
    st.write('Method:', method)
    if n_clusters is not None:
        st.write('n_clusters:', n_clusters)
    if eps is not None:
        st.write('eps:', eps)
    if min_samples is not None:
        st.write('min_samples:', min_samples)

    if run_button:
        unique_labels = set(labels)
        if len(unique_labels - {-1}) > 1:
            try:
                sil = silhouette_score(X_scaled, labels)
                st.metric('Silhouette Score', f'{sil:.3f}')
            except Exception:
                st.write('Silhouette score not available for selected labels')
        st.write('Cluster counts:')
        st.write(pd.Series(labels).value_counts().sort_index())

st.markdown('---')

st.subheader('Feature scatter')
fig2, ax2 = plt.subplots(figsize=(8,4))
sns.scatterplot(x=X[feature_x], y=X[feature_y], hue=labels if run_button else y, palette='tab10' if run_button else 'Set1', ax=ax2)
ax2.set_xlabel(feature_x)
ax2.set_ylabel(feature_y)
st.pyplot(fig2)

st.subheader('Data with Cluster Labels')
df_out = X.copy()
df_out['target'] = y
df_out['cluster'] = labels
st.dataframe(df_out.head(200))

csv = df_out.to_csv(index=False).encode('utf-8')
st.download_button('Download labeled data (CSV)', data=csv, file_name='wine_clusters.csv', mime='text/csv')

st.markdown('---')
st.write('Note: Fuzzy C-Means requires `scikit-fuzzy` (optional).')
