import pandas as pd
import streamlit as st
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score,confusion_matrix
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
import plotly.express as px

# --------------------------- Prétraitement ---------------------------
def normaliser_donnees(rfm):
    scaler = StandardScaler()
    rfm_norm = scaler.fit_transform(rfm)
    st.success("✅ Données RFM normalisées avec StandardScaler.")
    return rfm_norm

def renommer_clusters(clusters):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    return le.fit_transform(clusters) + 1  


# --------------------------- Clustering KMeans ---------------------------
def kmeans_clustering(rfm_norm, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(rfm_norm)
    
    st.success(f"✅ KMeans a été appliqué avec succès avec **{n_clusters}** clusters.")

    # Répartition des clients par cluster
    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    cluster_df = pd.DataFrame({
        'Cluster_ID': cluster_counts.index+1,
        'Nombre de clients': cluster_counts.values
    })
    st.dataframe(cluster_df)

    return clusters, kmeans

    
    # --------------------------- Clustering Hiérarchique CAH ---------------------------
def clustering_hierarchique(rfm_norm, n_clusters):
    linked = linkage(rfm_norm, method='ward')
    clusters = fcluster(linked, t=n_clusters, criterion='maxclust')
    st.success(f"✅ Clustering hiérarchique a été appliqué avec succès avec **{n_clusters}** clusters.")

    # Répartition des clients par cluster
    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    cluster_df = pd.DataFrame({
        'Cluster_ID': cluster_counts.index,
        'Nombre de clients': cluster_counts.values
    })
    st.dataframe(cluster_df)

    return clusters, linked


def afficher_dendrogramme(linked):
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(linked, truncate_mode='lastp', p=20, ax=ax)
    ax.set_title("Dendrogramme (Hiérarchique)")
    ax.set_xlabel("Index des échantillons")
    ax.set_ylabel("Distance")
    st.pyplot(fig)

def afficher_coude(rfm_norm):
    distortions = [KMeans(n_clusters=k, random_state=42).fit(rfm_norm).inertia_ for k in range(1, 11)]
    fig, ax = plt.subplots()
    ax.plot(range(1, 11), distortions, 'bx-')
    ax.set_xlabel('Nombre de clusters')
    ax.set_ylabel('Distorsion')
    ax.set_title('Méthode du coude')
    st.pyplot(fig)

def visualiser_clusters(rfm_norm, clusters_kmeans, clusters_hierarchique):
    pca = PCA(n_components=2)
    rfm_2d = pca.fit_transform(rfm_norm)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    sns.scatterplot(x=rfm_2d[:, 0], y=rfm_2d[:, 1], hue=clusters_kmeans, palette='tab10', ax=ax[0])
    ax[0].set_title("KMeans (PCA)")
    sns.scatterplot(x=rfm_2d[:, 0], y=rfm_2d[:, 1], hue=clusters_hierarchique, palette='tab10', ax=ax[1])
    ax[1].set_title("Hiérarchique (PCA)")
    st.pyplot(fig)
    st.markdown("""
    - Ces graphiques représentent les clients **projetés en 2D** après réduction de dimension via **PCA (Analyse en Composantes Principales)**.
    - Chaque **point** est un client, et chaque **couleur** représente un cluster attribué par l’algorithme (KMeans à gauche, CAH à droite).
    
    **Ce qu’on cherche à observer :**
    - 🎯 **Séparation nette entre les couleurs** ➜ bonne segmentation (les groupes sont distincts).
    - 🎨 **Clusters compacts** ➜ les clients d’un même groupe se ressemblent vraiment.
    - ⚠️ **Chevauchements ou formes diffuses** ➜ les clusters peuvent être moins fiables ou trop similaires.
    
    👉 Ces visualisations permettent de **valider visuellement** la qualité de la segmentation produite par les deux méthodes.
    """)



def ajouter_clusters(rfm, clusters_kmeans, clusters_hierarchique):
    rfm_copy = rfm.copy()
    rfm_copy['Cluster_KMeans'] = clusters_kmeans
    rfm_copy['Cluster_Hierarchique'] = clusters_hierarchique
    return rfm_copy


def afficher_comparaison_clusters(rfm_clustered):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Répartition - KMeans")
        st.metric("Nombre total de clusters (KMeans)", rfm_clustered['Cluster_KMeans'].nunique())
        st.bar_chart(rfm_clustered['Cluster_KMeans'].value_counts().sort_index())
    with col2:
        st.subheader("Répartition - CAH")
        st.metric("Nombre total de clusters (Hiérarchique)", rfm_clustered['Cluster_Hierarchique'].nunique())
        st.bar_chart(rfm_clustered['Cluster_Hierarchique'].value_counts().sort_index())


def comparer_scores_clusters(rfm_norm, clusters_kmeans, clusters_hierarchique):
    sil_kmeans = silhouette_score(rfm_norm, clusters_kmeans)
    sil_hier = silhouette_score(rfm_norm, clusters_hierarchique)
    db_kmeans = davies_bouldin_score(rfm_norm, clusters_kmeans)
    db_hier = davies_bouldin_score(rfm_norm, clusters_hierarchique)

    scores_df = pd.DataFrame({
        "Méthode": ["KMeans", "Hiérarchique"],
        "Silhouette Score (plus grand = mieux)": [sil_kmeans, sil_hier],
        "Davies-Bouldin Index (plus petit = mieux)": [db_kmeans, db_hier]
    })
    st.dataframe(scores_df)

    st.success(f"Meilleur Silhouette Score : {'KMeans' if sil_kmeans > sil_hier else 'Hiérarchique'}")
    st.success(f"Meilleur Davies-Bouldin Index : {'KMeans' if db_kmeans < db_hier else 'Hiérarchique'}")
    

def comparer_scores(rfm_norm, k_range):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    data = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42).fit(rfm_norm)
        sil_score = silhouette_score(rfm_norm, model.labels_)
        data.append((k, sil_score))
        
    return pd.DataFrame(data, columns=["n_clusters", "Silhouette"])

def afficher_extrait(rfm_clustered):
    st.dataframe(rfm_clustered.head(10))
    
def afficher_profils_clusters_kmeans(rfm_clustered):
    st.subheader("Profils des clusters - KMeans")
    col1, col2 = st.columns(2)
    with col1:
        profils = rfm_clustered.groupby('Cluster_KMeans')[['Recence', 'Frequence', 'Montant']].mean().round(2)
        st.dataframe(profils)
    with col2:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for i, col in enumerate(['Recence', 'Frequence', 'Montant']):
            sns.barplot(x=profils.index, y=profils[col], ax=axes[i])
            axes[i].set_title(f"Moyenne {col} par Cluster")
            axes[i].set_xlabel("Cluster")
            axes[i].set_ylabel(col)
        st.pyplot(fig)

def afficher_profils_clusters_cah(rfm_clustered):
    st.subheader("Profils des clusters - Hiérarchique")
    col1, col2 = st.columns(2)
    with col1:
        profils = rfm_clustered.groupby('Cluster_Hierarchique')[['Recence', 'Frequence', 'Montant']].mean().round(2)
        st.dataframe(profils)
    with col2:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for i, col in enumerate(['Recence', 'Frequence', 'Montant']):
            sns.barplot(x=profils.index, y=profils[col], ax=axes[i])
            axes[i].set_title(f"Moyenne {col} par Cluster")
            axes[i].set_xlabel("Cluster")
            axes[i].set_ylabel(col)
        st.pyplot(fig)