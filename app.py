import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.cluster.hierarchy import linkage
from modules import data_preparation, segmentation, association_rules_analysis

st.set_page_config(page_title="Projet Data Mining", layout="wide")

st.title("🧠 Projet de Data Mining : Analyse Client et Recommandations")
st.markdown("""
    **Ce projet s’appuie sur le jeu de données Online Retail issu de l’UCI Machine Learning Repository.**
    
    **Contient +500 000 transactions enregistrées (01/12/2010 - 09/12/2011).**
    
    **Objectifs clés :**
    - Une **préparation et une exploration approfondie** des données.
    - Une **segmentation client** basée sur la méthode RFM (Récence, Fréquence, Montant) avec clustering (K-Means/CAH).
    - Une **analyse des règles d'association** par l'algorithme Apriori pour détecter les produits fréquemment achetés ensemble et optimiser cross-selling et merchandising.
    - Des **recommandations business** basées sur ces analyses.
    """)

# Onglets principaux
tabs = st.tabs([
    "📂 Préparation et Exploration des Données",
    "📊 Segmentation Client",
    "🔗 Analyse des Règles d'Association"
])



# --------- 2. Données ---------
with tabs[0]:
    st.header("Étape 1 : Nettoyage et Exploration des Données")
    data = data_preparation.load_data("Online_Retail.csv")
    data = data_preparation.nettoyer_donnees(data)

    st.subheader("Aperçu des données après nettoyage")
    st.dataframe(data.head(10))
    
    st.header("Etape 2 : Analyse exploratoire")
    
    st.subheader("1. Statistiques descriptives :")
    st.write(data.describe())
    
    st.subheader("2. Visualisations des données:")
    st.subheader("Distribution des quantités par mois")
    col1, col2 = st.columns(2)
    
    with col1:
        
        data_preparation.visualiser_distribution_quantite_mois(data)
    with col2:
        st.markdown("""
    **Cette visualisation montre comment la quantité de produits vendus varie au fil des mois.**
    
    Elle permet d'identifier des tendances saisonnières ou des pics de ventes spécifiques à certains mois.
    - **La tendance globale est croissante :** les ventes augmentent au fil des mois, avec une nette accélération à partir de septembre.
    - **Novembre** est le mois record avec **669 051 unités** vendues, suivi de **décembre (599 678)** et **octobre (593 900)**.
    - **Février** a enregistré le plus faible total de ventes **(265 622)**, ce qui peut être lié à sa courte durée.
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 10 des produits par quantité vendue")
        data_preparation.visualiser_top_products(data)
        st.markdown("""
                - **Ciblage client :** Le succès des articles décoratifs ou "cadeaux" montre une clientèle axée sur l’esthétique et les événements (mariages, anniversaires, Noël).

                - **Stock à privilégier :** Les produits en tête de classement devraient être maintenus en stock prioritairement.

                - **Promotions :** Les articles entre la 5e et la 10e place pourraient bénéficier de campagnes pour booster leur visibilité.""")

    with col2:
        st.subheader("Top 10 des produits par revenu")
        data_preparation.top_products_revenu(data)
        st.markdown("""
        **Produit le plus rentable :**
                    
        **"PAPER CRAFT, LITTLE BIRDIE" (CA ~160k€)**, bien que absent du top volume.
                    
        **->** Les produits qui génèrent le plus de chiffre d'affaires, les produits artisanaux (ex: **"PAPER CRAFT"** ), ne sont pas nécessairement les best-sellers en volume.


        **Stratégies:** 
        Marge élevée ou prix unitaire haut, Associer les best-sellers en volume (ex: **"JUNIOR BAG"**) avec des produits haut de gamme ("PAPER CRAFT").""")
        
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 10 des pays par nombre de transactions")
        data_preparation.visualiser_top_countries(data)
        st.markdown("""
                - Le **Royaume-Uni** représente l’écrasante majorité des transactions **(≈350k)**.
                    
                - **L’Allemagne** et **la France** suivent avec des volumes significatifs **(≈80-90k)**.
                    
                - Les autres pays (**Espagne**, **Pays-Bas**, etc.) ont des volumes **5 à 10x inférieurs**.
                    
                **→** Malgré un faible CA, certains pays comme **l’Espagne** ou **la Belgique** ont un volume de transactions non négligeable.
                
            """)
    with col2:
        st.subheader("Top 10 des pays par revenu (chiffre d'affaires)")
        data_preparation.ca_country(data)
        st.markdown(""" 
                    
           - **Leader incontesté :** Le **Royaume-Uni** domine avec **7,3M€** de CA (≈ 95% du total).
                    
           - **Marchés secondaires :**
                - **Pays-Bas** (266k€), **Irlande** (229k€),**Allemagne** (209k€) et **France** (139k€) forment le peloton de tête.
                - Autres pays (**Australie**, **Espagne**, etc.) ont un CA marginal (<100k€).
           - Concentration extrême sur le marché domestique **(UK)**.
           - Cibler les **pays à fort CA** relatif (**Pays-Bas**, **Allemagne**) avec des campagnes localisées.
                    
            **→** Ces données révèlent un potentiel inexploité à l’étranger, malgré une forte dépendance au marché UK.
""")
    

    st.header("Etape 3 : Créations des données RFM")
    rfm = data_preparation.creer_variables_RFM(data)
    st.markdown("""
                - **Récence (Recence) :** Délai depuis le dernier achat (plus la valeur est faible, plus le client est récent).
                - **Fréquence (Frequence) :** Nombre d'achats sur une période donnée.
                - **Montant (Montant) :** Dépense moyenne par client.""")
    st.subheader("Aperçu des données RFM :")
    st.write(rfm.head())
    
    st.subheader("Visualisations des données RFM :")
    data_preparation.visualiser_histogram_rfm(rfm)
    cols = st.columns(3)
    cols[0].markdown("""
    **1. Distribution de la Récence (Jours depuis le dernier achat)**
                
    - **Pic à gauche :** La majorité des clients ont acheté **récemment (0-50 jours)**.
    - **Queue longue :** Une petite proportion de clients inactifs **(>200 jours sans achat).
    - **Insight :**
        - **Opportunité :** Cibler les clients récents **(≤50 jours)** avec des offres de fidélisation.
        - **Risque :** Clients inactifs **(>200 jours)** à relancer via campagnes de réactivation.""")

    cols[1].markdown("""           
    **2. Distribution de la Fréquence (Nombre de transactions)**
                
    - **Asymétrie extrême :**
        - La plupart des clients ont 1-5 commandes (pic à gauche).
        - Quelques clients très fréquents (>100 transactions) **→ "Super-clients"**.
    - **Insight :**
        - **Action :** Identifier les acheteurs ponctuels (1-2 achats) pour les convertir en clients réguliers (programmes de fidélité).
        - **VIPs :** Offrir des avantages exclusifs aux clients très fréquents.""")

    cols[2].markdown("""          
    **3. Distribution du Montant (Montant total dépensé)**
                
    - **Majorité low-spenders :** 80% des clients dépensent <100€ (pic à gauche).
    - **Few high-spenders :** Une minorité (queue droite) dépense >200€ **→ Source majeure de revenus**.
    - **Insight :**
        - **Stratégie :** 
            - **Upselling** pour les low-spenders (ex: bundles, offres premium).
            - **Personnalisation** pour les high-spenders (services sur-mesure, early access).
    """)
    st.subheader("Nuage de points RFM")
    data_preparation.visualiser_scatter_rfm(rfm)
    cols = st.columns(3)
    cols[0].markdown("""
    **1. Récence vs Fréquence**
    - **Tendance générale :**
        - Les clients **récemment actifs** (récence faible) ont une fréquence d'achat modérée (50-100 transactions).
        - Peu de clients combinent **haute fréquence ET récence élevée** (en haut à droite) **→** Les clients fréquents restent engagés.
    """)

    cols[1].markdown("""           
    **2. Fréquence vs Montant**
    - **Correlation positive :**
        - Les clients avec une **fréquence élevée** (≥100 achats) génèrent aussi un **montant total élevé (≥100k€)**.
        - Quelques outliers : Clients peu fréquents mais gros dépensiers (en haut à gauche) **→ Cibles pour upselling**.
                     
    Encourager les clients fréquents mais "low-spenders" (en bas à droite) à augmenter leur panier moyen.
    """)

    cols[2].markdown("""          
    **3. Récence vs Montant**
                
    - **Pas de lien évident :** 
        - Des clients **récents** (récence faible) apparaissent à la fois parmi les **petits et gros dépensiers**.
        - Les clients **inactifs** (récence >200 jours) montrent peu de gros montants **→ Perte de revenus potentiels**.
                     
        Les clients récents avec un montant élevé sont des **nouvelles cibles prioritaires**.

    """)
    



# --------- 3. Segmentation ---------
with tabs[1]:
    st.header("Etape 1 : Normalisation des données RFM")
    rfm_reset = rfm.reset_index()
    rfm_norm = segmentation.normaliser_donnees(rfm_reset[['Recence', 'Frequence', 'Montant']])
        
    st.header("Etape 2 : Choix de nombre optimal de clusters")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("1.KMeans - Méthode du Coude")
        fig_coude = segmentation.afficher_coude(rfm_norm)
    with col2:
        st.subheader("2.CAH - Dendrogramme")
        linked = linkage(rfm_norm, method='ward')
        segmentation.afficher_dendrogramme(linked)

    with col3:
        st.subheader("3. Comparaison des scores")
        k_range = range(2, 11)
        df_scores = segmentation.comparer_scores(rfm_norm,k_range)
        st.dataframe(df_scores)
        
    st.markdown("### ✅ Suggestion du nombre optimal de clusters")
    st.markdown("""
    - Le **score de silhouette maximal** est atteint pour **k = 5** (`0.6171`) → séparation relativement bonne.
    - Toutefois, la **méthode du coude** montre un changement notable de pente autour de **k = 4**, indiquant un compromis acceptable entre inertie et interprétabilité.
    - **Dendrogramme (CAH)** : une coupure visuelle autour de la hauteur **50–60** révèle clairement **4 à 5 grands clusters bien distincts**, confirmant les résultats des autres méthodes.


    - k = 4 ou k = 5 sont les meilleurs choix :
        - k = 4 : plus simple à interpréter, cohérent avec le coude
        - k = 5 : meilleure séparation des groupes (meilleur silhouette score)
    - 🎯 **Conclusion : `k = 5` est recommandé** pour une segmentation plus précise, tandis que `k = 4`reste une alternative plus simple et robuste.
    """)

    
    st.header("Etape 3 : Application de clustering")
    st.subheader("1. Choix du nombre de clusters")
    n_clusters = st.slider("Choisissez le nombre de clusters", 2, 10, 5)
    
    st.header("")
    st.header("")
    st.header("")
    st.header("")
    st.header("")
    st.header("")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("2. Application de KMeans")
        clusters_kmeans, _ = segmentation.kmeans_clustering(rfm_norm, n_clusters=n_clusters)
        clusters_kmeans = segmentation.renommer_clusters(clusters_kmeans)
        rfm_reset['Cluster_KMeans'] = clusters_kmeans
    with col2:
        st.subheader("3. Application de CAH")
        clusters_cah, linked = segmentation.clustering_hierarchique(rfm_norm, n_clusters=n_clusters)
        rfm_reset['Cluster_CAH'] = clusters_cah
        rfm_clustered = segmentation.ajouter_clusters(rfm, clusters_kmeans, clusters_cah)
        
        
        
    st.header("Etape 4 : Analyse des clusters")
    st.subheader("1. Visualisation des clusters")
    segmentation.afficher_extrait(rfm_clustered)
    
    st.subheader("2. Répartition des clusters")
    segmentation.afficher_comparaison_clusters(rfm_clustered)
    cols = st.columns(2)
    cols[0].markdown("""
                La méthode KMeans produit une répartition inégale des données entre les clusters.
                
                - Deux clusters dominants (cluster 1 et 2) regroupent la majorité des observations, ce qui suggère une concentration importante des données dans ces groupes.
                - Les trois autres clusters sont plus petits, avec des tailles décroissantes (2 000, 1 500, 1 000). Cela indique la présence de sous-groupes moins peuplés, possiblement des niches ou des outliers regroupés.
                - **KMeans :** Utile pour identifier des groupes dominants et des outliers, mais peut négliger des sous-groupes intermédiaires.""")
    
    cols[1].markdown("""
                - La CAH montre également une répartition inégale, mais avec des écarts moins marqués que KMeans.
                - Les deux premiers clusters restent les plus grands.
                - Le troisième cluster est plus important (2 500 contre 2 000 pour KMeans), ce qui pourrait refléter une agrégation différente des données par la méthode hiérarchique.
                - La CAH tend à créer des clusters plus équilibrés pour les groupes secondaires, ce qui peut indiquer une meilleure sensibilité aux nuances dans les données.""")
    
    st.subheader("3. Analyse des profils de clusters")
    segmentation.afficher_profils_clusters_kmeans(rfm_clustered)
    cols = st.columns(3)
    cols[0].markdown("""
                - Certains clusters (3, 4, 5) ont une **récence très faible** (~15 jours **→** clients **réactifs**) 
                     
                     **→ Cibles pour des campagnes de fidélisation immédiate.**
                - D’autres ont une **récence élevée** (ex. >200 jours **→** clients **inactifs** ou **dormants**) 
                     
                     **→ Besoin de relance (promotions, emails reactivation).**
                    """)
    
    cols[1].markdown("""
                - Certains clusters (3, 4, 5) montrent une **fréquence très élevée** (ex. >20 achats → clients **fidèles** ou **B2B**) 
                     
                     **→ Programmes **VIP** ou **abonnements**.**
                - D’autres ont une **fréquence faible** (ex. <5 achats **→** clients **occasionnels**) 
                     
                     **→ Stimuler la répétition d'achat (bundles, offres incitatives).**
        """)
    
    cols[2].markdown("""
                    - Certains clusters (3, 4, 5) ont un **montant très élevé** (ex. pic à droite → **gros dépensiers**) 
                     
                     **→ Offres haut de gamme ou services personnalisés.**
                    - D’autres ont un montant modéré ou faible (ex. gauche du graphique → clients **économiques**) 
                     
                     **→ Focus sur le volume (ex. ventes groupées).**
        """)
    segmentation.afficher_profils_clusters_cah(rfm_clustered)

    st.header("Etape 5 : Évaluation des méthodes de clustering")
    st.subheader(" 1.Scores d'évaluation - silhouette et Davies-Bouldin")
    segmentation.comparer_scores_clusters(rfm_norm, clusters_kmeans, clusters_cah)



# --------- 4. Association ---------
with tabs[2]:
    st.header("🧹 Étape 1 : Préparation des données")
    st.markdown("""
    Dans cette première étape, nous préparons les données pour l’analyse des règles d’association :
    
    - ✅ **Transformation des données en format transactionnel** (chaque ligne = un client et les articles achetés).
    - ✅ **Encodage binaire des articles** (0/1 selon l’achat de chaque article).
    """)
    
    df_transactions = association_rules_analysis.preparer_transactions(data)
    
    
    st.header("🧠 Étape 2 : Génération des Règles d'Association")
    st.subheader("⚙️ 1. Choix des paramètres pour l'algorithme Apriori")
    st.markdown("""
    Définissez les seuils pour générer des règles pertinentes.
    """)
        
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - **Support minimum :** Fréquence d’apparition des items dans les transactions totales.
        """)
    with col2:  
        min_support = st.slider("Support minimum (%)", 1, 50, 1) / 100
        
        
        
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - **Confiance :** Probabilité que l’item B soit acheté si A l’est.    """)
    with col2:
        min_confidence = st.slider("Confiance minimum (%)", 10, 100, 50) / 100
        
        
        
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - **Lift :** Indique l’importance de la règle ( >=1 corrélation positive).
        """)
    with col2:
        min_lift = st.slider("Lift minimum", 1.0, 5.0, 1.0)
        
        

    st.subheader("📈 2. Génération et affichage des règles d'associations")
    with st.spinner("Génération en cours..."):
        rules = association_rules_analysis.appliquer_apriori(df_transactions, min_support, min_confidence, min_lift)
        
        if not rules.empty:
            st.success(f"{len(rules)} règles générées ✅")
            
            # Convertir frozenset en chaînes lisibles
            rules_affiche = rules.copy()
            rules_affiche['antecedents'] = rules_affiche['antecedents'].apply(lambda x: ', '.join(sorted(x)))
            rules_affiche['consequents'] = rules_affiche['consequents'].apply(lambda x: ', '.join(sorted(x)))


            st.markdown("### 📋 Tableau des règles principales")
            st.dataframe(rules_affiche[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
            
            st.header("Etape 3 : Visualisation")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🔝 1. Règles les plus fortes")
                association_rules_analysis.plot_top_rules(rules)   # bar chart des règles les + fortes
                st.markdown("""
                        **1. Règles avec les lifts les plus élevés (~60) :**

                        - **REGENCY TEA PLATE GREEN → REGENCY TEA PLATE PINK**

                        - **REGENCY TEA PLATE PINK → REGENCY TEA PLATE GREEN**
                            
                        **→** Ces deux produits sont très **complémentaires**.

                        **2. Enchaînements dans la même gamme :**

                        - Plusieurs règles concernent les produits **POPPY'S PLAYHOUSE** (kitchen, bedroom, livingroom)
                            
                        **→** Cela montre un **comportement d’achat groupé** dans la même gamme de produits.

                        **3. Compléments naturels :**

                        - **REGENCY MILK JUG PINK → REGENCY SUGAR BOWL GREEN**

                        - **REGENCY SUGAR BOWL GREEN → REGENCY MILK JUG PINK**
                            
                        **→** Les clients achètent ensemble des **accessoires de vaisselle assortis**.""")
            with col2:
                st.subheader("🛒 2. Articles les plus fréquents")
                association_rules_analysis.top_frequent_items(rules) # bar chart des items les + fréquents
                st.markdown("""
                        **1. Domination des "Lunch Bags" :**

                        - Les 7 premières positions sont occupées par des variantes de sacs à déjeuner.

                        **→** Cela montre que les **Lunch Bags sont des produits centraux** dans les paniers d’achat. Ils sont souvent achetés avec d'autres articles.

                        **2. Popularité des produits “REGENCY TEACUP AND SAUCER” :**

                        - Les 3 dernières positions concernent différentes versions de tasses avec soucoupes de la gamme **REGENCY** (rose, verte, à fleurs).

                        **→** Cela souligne un autre **comportement d’achat groupé** autour des articles de vaisselle assortis.""")
                
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📍  3. Nuage de règles (Support vs Confiance)")
                association_rules_analysis.scatter_plot_support_confiance(rules)  # scatter plot des règles
                st.markdown("""
                        **1. Concentration de points à faible support (< 0.015) :** La plupart des règles apparaissent rarement dans les données.

                        **2. Distribution de la confiance :**

                        - Quelques règles ont une confiance élevée (>0.8), indiquant qu'elles sont très fiables.

                        **3. Taille/couleur des bulles (Lift) :**

                        - Les plus grosses bulles (rouges foncées) indiquent des règles très intéressantes (**lift élevé (>50)**).
                        - Ces bulles sont souvent associées à un **support faible mais une confiance élevée** : ce sont des règles rares mais très fortes.
                            
                       **→ Les meilleures règles** (intéressantes et fiables) sont celles avec :
                        - **Haute confiance**
                        - **Lift élevé**
                        """)                    
            with col2:
                st.subheader("📍 4. Nuage de règles (Lift vs Confiance)")
                association_rules_analysis.scatter_plot_lift_confiance(rules)
                st.markdown("""
                           **1. Règles avec lift élevé (> 30) :**

                            - Elles ne sont pas les plus nombreuses.
                            - La plupart ont une confiance modérée (entre 0.55 et 0.70).
                            - Elles ont un support faible**.

                            **2. Grosses bulles (support élevé) :**

                            - Elles ne sont pas les plus nombreuses.
                            - La plupart ont une **confiance modérée** (entre 0.55 et 0.70).
                            - Elles ont un **support faible**.

                            **2. Grosses bulles (support élevé) :**

                            - Elles ont un **lift entre 5 et 20**.
                            - Certaines ont une bonne confiance (>0.75), ce qui les rend intéressantes **par leur fréquence et leur fiabilité.**

                            **3. Bulle idéale ?**

                            - Une règle avec **lift > 20, confiance > 0.75, support élevé (grosse bulle)** serait idéale.
                            - On voit quelques cas vers (0.8, 20) avec des grosses bulles : très bonnes candidates.

                            """)
                
            st.subheader("🔥 5. Carte thermique des lifts")
            association_rules_analysis.heatmap_lift(rules) # heatmap des lifts
            st.markdown("""
                        **1. Les 2 premières règles (REGENCY TEA PLATE PINK ↔ GREEN) :**

                        - **Lift = 61.90, Confiance élevée (0.90 / 0.75)**
                        ➤ Très forte association, très fiable malgré support faible.

                        **2. Les règles POPPY'S PLAYHOUSE :**

                        - Confiance entre 0.59 et 0.87
                        - Lift entre 50.74 et 53.85
                        ➤ Très bonnes corrélations entre différentes pièces d’un même univers produit.

                       **3.  Autres règles (TEA PLATE ROSES, MILK JUG, SUGAR BOWL) :**

                        - Aussi très élevées en lift (>49)
                        - Indiquent des achats groupés typiques de produits assortis.
                        
                        **Conclusion** 
                        - Toutes ces règles sont **très pertinentes (lift élevé)**.
                        - **Confiance > 0.75** = fiable ➤ règles à prioriser pour des recommandations.
                        - Bien que **le support soit faible**, ces règles méritent attention pour **ventes croisées ciblées**.""")
            
            st.subheader("6. Produits fréquemment achetés ensemble")
            col1, col2 = st.columns(2)
            with col1:
                association_rules_analysis.frequent_items_together(rules, top_n=10, filtrer_complexes=True)     
            with col2:
                      
                st.markdown("""
                        
                **Exemples clés du graphe :**
                                
                **1. REGENCY TEA PLATE PINK → REGENCY TEA PLATE GREEN**

                - **Lift : 61.90** ➤ très forte corrélation
                - Ces deux assiettes sont très souvent achetées ensemble.

                **2. REGENCY MILK JUG PINK → REGENCY SUGAR BOWL GREEN**

                - **Lift : 52.37**
                - Ensemble cohérent pour le thé/petit-déjeuner.
                
                **3. POPPY'S PLAYHOUSE BEDROOM → POPPY'S PLAYHOUSE LIVINGROOM**

                - **Lift : 47.71**
                - Les produits d'une même gamme sont souvent achetés ensemble.        """)
            
            st.subheader("💡 7. Simulateur de recommandation")

            association_rules_analysis.simulateur_recommandation(rules) # simulateur de recommandation
        else:
            st.warning("⚠️ Aucune règle trouvée avec ces paramètres.")
            






