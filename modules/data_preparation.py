import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import streamlit as st
import plotly.express as px

# Chargement du jeu de données
@st.cache_data
def load_data(filepath):
    return pd.read_csv(filepath, encoding='latin1')

def nettoyer_donnees(df):
    st.info(f"✅ Données chargées avec succès.  \n**Nombre total de lignes : {df.shape[0]}**")
    st.markdown("""
    **Les étapes de nettoyage :**
""")
    df = df.dropna(subset=['CustomerID'])
    st.markdown("""
    - **Suppression des valeurs manquantes,** en particulier celles concernant le code client (CustomerID) et la description du produit (Description).
""")
    st.success(f"1. 🧹 Suppression des lignes sans identifiant client.  \n**Nombre de lignes restantes : {df.shape[0]}**")
    
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    st.markdown("""
    - **Élimination des lignes avec des quantités négatives ou nulles**, considérées comme transaction annulée.
    """)
    st.success(f"2. 📦 Filtrage des produits retournés.  \n**Nombres de lignes valides conservées : {df.shape[0]}**")
    
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    st.markdown("""
    - **Conversion des types de données**, notamment (InvoiceDate) vers un format (datetime).
""")
    st.success("3. 📅 Les dates des transactions ont été converties **au bon format datetime**.")
    
    return df

    
def visualiser_distribution_quantite_mois(df):
    # Création de la colonne 'Month' à partir de 'InvoiceDate'

    df['Month'] = df['InvoiceDate'].dt.month

    # Calcul du total des quantités par mois
    monthly_quantities = df.groupby('Month')['Quantity'].sum()

    # Création de l'histogramme
    plt.figure(figsize=(10, 6))
    sns.barplot(x=monthly_quantities.index, y=monthly_quantities.values, palette='Blues_d')
    plt.title("Total des quantités vendues par mois")
    plt.xlabel("Mois")
    plt.ylabel("Quantité totale vendue")
    plt.xticks(ticks=range(0, 12), labels=[
        'Janv', 'Fév', 'Mars', 'Avr', 'Mai', 'Juin',
        'Juil', 'Août', 'Sept', 'Oct', 'Nov', 'Déc'
    ])

    # Affichage des valeurs sur chaque barre
    for i, value in enumerate(monthly_quantities.values):
        plt.text(i, value + 10, f'{value:.0f}', ha='center', fontsize=12, color='black')

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()
    
    

    
def visualiser_top_products(df):
    #  Filtrer les lignes où Description != 'POSTAGE'
    df_filtered = df[df['Description'].str.upper() != 'POSTAGE']

    #  Création de la variable 'TotalPrice'
    df_filtered['TotalPrice'] = df_filtered['Quantity'] * df_filtered['UnitPrice']

    #  Top 10 produits les plus vendus 
    top_products = df_filtered['Description'].value_counts().head(10)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=top_products.values, y=top_products.index, palette='viridis', ax=ax)
    ax.set_title("Top 10 produits les plus vendus")
    ax.set_xlabel("Quantité vendue")
    ax.set_ylabel("Produit")
    st.pyplot(fig)
    
    
def visualiser_top_countries(df):
    country_counts = df['Country'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=country_counts.index, y=country_counts.values, palette='rocket',ax=ax)
    plt.title("Top 10 pays par nombre de transactions")
    plt.xlabel("Pays")
    plt.ylabel("Nombre de transactions")
    plt.xticks(rotation=45)

    # Annotations sur les barres
    for i, v in enumerate(country_counts.values):
        plt.text(i, v + 20, str(v), ha='center')

    plt.tight_layout()
    st.pyplot(fig)
    
    
def heatmap_quantite_par_jour_mois(df):
    # Extraire les jours de la semaine et les mois
    df['JourSemaine'] = df['InvoiceDate'].dt.day_name()
    df['Mois'] = df['InvoiceDate'].dt.month

    # Groupement des données : Somme des quantités vendues par jour de la semaine et mois
    heatmap_data = df.groupby(['JourSemaine', 'Mois'])['Quantity'].sum().unstack().fillna(0)

    # Réordonner les jours pour que ce soit lisible
    ordre_jours = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(ordre_jours)

    # Créer des labels de mois en français (ou en abrégé)
    noms_mois = ['Janv', 'Fév', 'Mars', 'Avr', 'Mai', 'Juin',
                 'Juil', 'Août', 'Sept', 'Oct', 'Nov', 'Déc']
    heatmap_data.columns = [noms_mois[m-1] for m in heatmap_data.columns]

    # Affichage de la heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap='YlOrBr', annot=True, fmt='.0f', linewidths=.5, ax=ax)
    ax.set_title("🗓️ Heatmap des ventes par jour de la semaine et mois")
    st.pyplot(fig)





def top_products_revenu(df):
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    top_rev = df.groupby('Description')['TotalPrice'].sum().sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=top_rev.values, y=top_rev.index, palette='magma', ax=ax)
    ax.set_title("Top 10 produits par chiffre d'affaires généré")
    ax.set_xlabel("Revenu (€)")
    ax.set_ylabel("Produit")
    st.pyplot(fig)


def ca_country(df):
    df = df[df['Country'].notnull()]

    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    ventes_par_pays = df.groupby('Country')['TotalPrice'].sum().reset_index()

    # Trier par chiffre d'affaires décroissant et garder le top 10
    ventes_par_pays = ventes_par_pays.sort_values(by='TotalPrice', ascending=False).head(10)

    # Arrondir les valeurs pour les rendre plus lisibles
    ventes_par_pays['TotalPrice'] = ventes_par_pays['TotalPrice'].round(-3)

    # Création du graphique à barres horizontales
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='TotalPrice', y='Country', data=ventes_par_pays, palette='magma', ax=ax)

    # Ajouter les valeurs sur les barres
    for p in ax.patches:
        ax.annotate(f'€{p.get_width():,.0f}', 
                    (p.get_width() + 500, p.get_y() + p.get_height() / 2),
                    ha='left', va='center', fontsize=10, color='black')

    ax.set_title("Top 10 des pays par chiffre d'affaires")
    ax.set_xlabel("Revenu (€)")
    ax.set_ylabel("Pays")

    st.pyplot(fig)
    


def creer_variables_RFM(df):
    date_reference = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    recence = df.groupby('CustomerID')['InvoiceDate'].max().apply(lambda x: (date_reference - x).days)
    frequence = df.groupby('CustomerID')['InvoiceNo'].nunique()
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    montant = df.groupby('CustomerID')['TotalAmount'].sum()
    
    rfm = pd.DataFrame({
        'Recence': recence,
        'Frequence': frequence,
        'Montant': montant
    })
    return rfm
    
def visualiser_histogram_rfm(rfm):    
    # Histos RFM
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(rfm['Recence'], bins=30, kde=True, color='skyblue', ax=axs[0])
    axs[0].set_title('Distribution de la Récence')
    axs[0].set_xlabel('Jours depuis le dernier achat')

    sns.histplot(rfm['Frequence'], bins=30, kde=True, color='lightgreen', ax=axs[1])
    axs[1].set_title('Distribution de la Fréquence')
    axs[1].set_xlabel('Nombre de transactions')

    sns.histplot(rfm['Montant'], bins=30, kde=True, color='salmon', ax=axs[2])
    axs[2].set_title('Distribution du Montant')
    axs[2].set_xlabel('Montant total dépensé (€)')

    st.pyplot(fig)
    
def visualiser_scatter_rfm(rfm):
    # Scatter plots RFM
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    sns.scatterplot(x='Recence', y='Montant', data=rfm, color='tomato', ax=axs[0])
    axs[0].set_title('Récence vs Montant')

    sns.scatterplot(x='Frequence', y='Montant', data=rfm, color='teal', ax=axs[1])
    axs[1].set_title('Fréquence vs Montant')

    sns.scatterplot(x='Recence', y='Frequence', data=rfm, color='purple', ax=axs[2])
    axs[2].set_title('Récence vs Fréquence')

    st.pyplot(fig)
    



    
    


