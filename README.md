# 🛍️ Projet Data Mining – Segmentation Client & Recommandation Produit

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)

## 📌 Présentation du projet

Ce projet vise à analyser le comportement des clients à partir du dataset [Online Retail (UCI Repository)](https://archive.ics.uci.edu/ml/datasets/Online+Retail).  
Il combine des techniques de **nettoyage de données**, **segmentation RFM**, **clustering (KMeans & CAH)** et **règles d’association** pour générer des recommandations produits exploitables.

---

## 🎯 Objectifs

- Segmenter les clients selon leur comportement d’achat (Récence, Fréquence, Montant)
- Identifier des profils clients via des algorithmes de clustering
- Détecter des associations produits pour optimiser le cross-selling
- Proposer une interface interactive pour explorer les résultats

---

## 🛠️ Stack technique

- **Python** : Pandas, NumPy, Scikit-learn, mlxtend, Seaborn, Matplotlib, Plotly
- **Clustering** : KMeans, Clustering hiérarchique (CAH)
- **Recommandation** : Apriori
- **Visualisation** : Streamlit
- **Source de données** : Online Retail (format Excel)

---

## 📊 Fonctionnalités principales

- Nettoyage et préparation des données (valeurs manquantes, annulations, conversions)
- Segmentation RFM et scoring client
- Comparaison des clusters : KMeans vs CAH (silhouette, dendrogramme)
- Extraction de règles d’association avec support, confiance et lift
- Heatmap des règles les plus liftées
- Application Streamlit pour l’exploration interactive

---

## 📂 Structure du projet
```
├── modules/ 
    ├── data_preparation.py 
    ├── segmentation.py 
    └── association_rules_analysis.py 
├── app.py 
├── OnlineRetail.csv
├── README.md
├── requirements.txt
```
---

## 🚀 Lancer le projet

1. Cloner le dépôt  
   ```bash
   git clone https://github.com/ritejth/data_mining_project.git
   ```

3. Installer les dépendances  
   ```bash
   pip install -r requirements.txt
   ```

5. Lancer l’application Streamlit  
   ```bash
   streamlit run app.py
   ```

---

## 📬 Contact

**Ritej Touhami**  
Étudiante en Master Professionnel en Ingénierie des Systèmes d’Information & Data Science 
📧 ritejtouhami@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/ritejtouhami)
