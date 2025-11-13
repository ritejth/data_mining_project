import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
import networkx as nx
from matplotlib.cm import ScalarMappable
from collections import Counter

# 1. Préparer les données transactionnelles
def preparer_transactions(df):
    transactions = df.groupby('InvoiceNo')['Description'].apply(list).values.tolist()    
            
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    st.success(f"✅ {len(transactions)} transactions préparées.")
    return df_encoded

# 2. Appliquer l'algorithme Apriori
@st.cache_data
def appliquer_apriori(df_encoded, min_support, min_confidence, min_lift):
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    
    if frequent_itemsets.empty:
        st.warning("⚠️ Aucun itemset fréquent trouvé. Réduisez le support minimal.")
        return pd.DataFrame()
    
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
    rules = rules[rules['confidence'] >= min_confidence]
    st.success(f"✅ {len(frequent_itemsets)} itemsets fréquents trouvés (support ≥ {min_support})")

    return rules.sort_values(by='lift', ascending=False)



# 5. Visualisation des meilleures règles (bar chart)
def plot_top_rules(rules, metric='lift', top_n=10):
    if rules.empty:
        return

    top_rules = rules.head(top_n)
    top_rules['rule'] = top_rules.apply(
        lambda row: f"{', '.join(list(row['antecedents']))} → {', '.join(list(row['consequents']))}", axis=1
    )
    
    # st.subheader("Top 10 règles par lift")
    
    plt.figure(figsize=(8, 8))
    sns.barplot(data=top_rules, x=metric, y='rule', palette='viridis')
    plt.xlabel(metric.capitalize())
    plt.ylabel("Règle")
    plt.title(f"Top {top_n} règles par {metric}")
    st.pyplot(plt.gcf())
    plt.clf()
    
    
def top_frequent_items(rules):  
    # Extraire tous les items des antécédents
    items = rules['antecedents'].apply(lambda x: list(x)).sum()
    item_counts = Counter(items)

    top_items = dict(sorted(item_counts.items(), key=lambda x: x[1], reverse=True)[:10])

    fig, ax = plt.subplots()
    ax.barh(list(top_items.keys()), list(top_items.values()))
    ax.invert_yaxis()
    plt.title("Top 10 items les plus fréquents dans les règles")
    # st.subheader("📦 Top 10 items les plus fréquents dans les règles")
    st.pyplot(fig)
    
    
def scatter_plot_support_confiance(rules):
    # Créer le plot
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(
        data=rules,
        x='support',
        y='confidence',
        size='lift',
        hue='lift',
        palette='coolwarm',
        sizes=(40, 400),
        alpha=0.7,
        edgecolor='black'
    )

    plt.title('Support vs Confidence (taille = Lift)')
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.grid(True)

    # Afficher avec Streamlit
    st.pyplot(plt)
    
def scatter_plot_lift_confiance(rules):
    # Bubble chart
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=rules,
        x='confidence',
        y='lift',
        size='support',
        hue='support',
        sizes=(20, 500),
        alpha=0.7,
        palette='cool',
        ax=ax3
    )
    ax3.set_title("Lift vs Confiance (taille = Support)")
    ax3.set_xlabel("Confiance")
    ax3.set_ylabel("Lift")
    st.pyplot(fig3)


def heatmap_lift(rules):
    if len(rules) >= 10:
        top_rules = rules.sort_values('lift', ascending=False).head(10)
        top_rules['rule'] = top_rules.apply(lambda row: f"{', '.join(list(row['antecedents']))} → {', '.join(list(row['consequents']))}", axis=1)
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            top_rules[['support', 'confidence', 'lift']].set_index(top_rules['rule']),
            annot=True, cmap='YlGnBu', fmt=".2f", ax=ax4
        )
        ax4.set_title("Top 10 Règles – Metrics Heatmap")
        ax4.set_ylabel('Règle')
        ax4.set_xlabel('Métrique')
        st.pyplot(fig4)


def frequent_items_together(rules, top_n=10, filtrer_complexes=True):

    if filtrer_complexes:
        rules = rules[rules['antecedents'].apply(lambda x: len(x) == 1) & rules['consequents'].apply(lambda x: len(x) == 1)]

    top_rules = rules.sort_values(by='lift', ascending=False).head(top_n).copy()
    G = nx.DiGraph()
    for _, row in top_rules.iterrows():
        antecedent = next(iter(row['antecedents']))
        consequent = next(iter(row['consequents']))
        G.add_edge(antecedent, consequent, weight=row['lift'])

    fig, ax = plt.subplots(figsize=(15, 10))
    pos = nx.kamada_kawai_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightyellow', edgecolors='black', ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=10, width=2, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', font_color='black', ax=ax)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    edge_labels = {edge: f"Lift: {lift:.2f}" for edge, lift in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_color='red', ax=ax)
    ax.axis('off')
    st.pyplot(fig)


def simulateur_recommandation(rules):
    if rules.empty:
        st.warning("❌ Aucune règle à utiliser pour les recommandations.")
        return

    tous_produits = sorted(set([item for sublist in rules['antecedents'] for item in sublist]))
    produit = st.selectbox("🎯 Choisissez un produit", tous_produits)

    st.subheader(f"📦 Produits fréquemment achetés avec : {produit}")
    associes = rules[rules['antecedents'].apply(lambda x: produit in x)]
    associes = associes.sort_values(by='lift', ascending=False).head(5)
    associes = associes.drop_duplicates(subset=['consequents'])

    if associes.empty:
        st.info("Aucune recommandation disponible pour ce produit.")
    else:
        for _, row in associes.iterrows():
            cibles = ', '.join(list(row['consequents']))
            st.markdown(f"👉 **{cibles.strip()}** (confiance: {row['confidence']:.0%}, lift: {row['lift']:.2f})")