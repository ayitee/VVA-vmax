## F1 Race Simulator — Quick start

This repository contains a Streamlit app to simulate F1 race results using local CSV data and a saved model.

Prerequisites
- Python 3.10+ (3.11/3.12/3.13 should work)
- git (optional)

Minimal setup

1. Create and activate a virtual environment (zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Upgrade pip and install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Start the app with Streamlit:

```bash
streamlit run app.py
```

Notes
- The app expects CSV files under the `data/` directory (these are included in the repo). If a required file is missing the app will try to degrade gracefully but some features may not work.
- A trained model file named `model_f1_rf.joblib` is expected at the repository root. If it is missing the app will show a warning and most prediction features will be disabled.
- If you reorganize files into a package layout, update imports such as `images.py` accordingly.

Troubleshooting
- If Streamlit complains about duplicate widget IDs, add explicit `key=` parameters to widgets that appear multiple times in the UI.
- If you see UI layout issues, try a hard refresh in the browser and restart the Streamlit process.

That's it — the three commands above are all you need to get the app running locally.
# Victory Vision Analytics 🏎️📊

## 🎯 Objectif
Prédire les résultats des courses de Formule 1 en intégrant l’influence de la météo.  
L’idée est d’analyser si et comment les conditions météorologiques peuvent :  
- Favoriser certaines équipes ou pilotes  
- Resserrer les écarts de temps  
- Donner plus de chances aux « petites » équipes  
- Introduire un facteur d’incertitude dans les résultats  

---

## 📦 Données utilisées
- `circuits.csv`  
- `constructor_results.csv`  
- `constructor_standings.csv`  
- `drivers.csv`, `driver_standings.csv`, `lap_times.csv`, `pit_stops.csv`, `qualifying.csv`, `races.csv`, `results.csv`, `seasons.csv`, `sprint_results.csv`, `status.csv`  
- `weather_features_v4.parquet` (météo quotidienne agrégée)  
- **+ autres datasets via Kaggle.com**  

---

## 🛠️ Outils
- **Pandas** → chargement, nettoyage et transformation des données  
- **Pyarrow** → lecture des fichiers `.parquet`  
- **Scikit-learn** → pipelines, classification (RandomForest, Logistic Regression)  
- **Seaborn / Matplotlib** → graphiques exploratoires  
- **Numpy** → simulation Monte Carlo  
- **Streamlit** → application web interactive  

---

## ✅ Livrables
1. **Notebook d’analyse et de modélisation** (`f1_race_prediction.ipynb`)  
   - Intégration robuste de la météo (`weather_features_v4.parquet`)  
   - Feature engineering, entraînement, évaluation, importance des variables  
   - Export du modèle: `model_f1_rf.joblib`  

2. **Application Streamlit** (`app.py`)  
   - Sélection de Saison, Grand Prix (Circuit), Écurie, Pilote — le numéro de voiture est affiché dans le libellé du pilote (UI en français)  
   - Mode « Pilote »: probabilité de marquer des points  
   - Mode « Grand Prix »: classement estimé (heuristique)  
   - Mode « Simulation (bêta) »: Monte Carlo des classements avec DNF/incidents et variabilité, incluant:
     - Proba ≥ 1 point (clarifie l'ancien « proba points »)
     - Proba ≥ X points (seuil personnalisable dans l'UI)

---

## 🚀 Démarrage rapide

1) Créer un environnement et installer les dépendances

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Entraîner le modèle (dans le notebook)
- Ouvrir `f1_race_prediction.ipynb`, exécuter les cellules jusqu’à l’export `model_f1_rf.joblib`.

3) Lancer l’application web

```bash
streamlit run app.py
```

Puis ouvrir l’URL locale indiquée (ex: http://localhost:8501).

Notes:
- Si le fichier `model_f1_rf.joblib` est manquant, l’app demande de l’entraîner et de l’exporter via le notebook.
- Les paramètres météo peuvent être ajustés dans l’UI (ou surchargés pour l’ensemble du GP dans les modes classements/simulation).

---

## ✂️ Pruning (réduction) du dataset

Pour accélérer l’app et réduire la taille des données, vous pouvez créer un dossier de données « pruné » :

1) Lancer le script (mode simulation « dry-run » d’abord)

```bash
python tools/prune_dataset.py --data-dir data --out-dir data_pruned --min-year 2001 --dry-run
```

2) Exécuter réellement la réduction (écrit les CSV filtrés dans `data_pruned`)

```bash
python tools/prune_dataset.py --data-dir data --out-dir data_pruned --min-year 2001
```

Optionnel: garder uniquement les pilotes actifs dans une saison donnée (ex: 2024)

```bash
python tools/prune_dataset.py --data-dir data --out-dir data_pruned --min-year 2001 --keep-season 2024
```

3) Pointer l’app sur le dossier pruné (sans changer le code)

```bash
export VVA_DATA_DIR=$(pwd)/data_pruned
streamlit run app.py
```

Sécurité et remarques:
- Le script ne modifie pas les fichiers originaux; il écrit dans `--out-dir`.
- Les fichiers non-CSV (ex: `.parquet`) sont copiés tels quels.
