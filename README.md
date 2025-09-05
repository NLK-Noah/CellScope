# CellScope
# DATASET
```bash
https://www.kaggle.com/datasets/paultimothymooney/blood-cells
```
# Arborescense 
```bash
CellScope/
├─ data/                   # Datasets d’images de cellules (train/test/validation)
│   ├─ train/
│   ├─ test/
│   └─ validation/
├─ models/                 # Modèles ML/IA enregistrés
│   └─ cell_classifier.pt
├─ notebook/               # Notebooks pour exploration/entraînement
│   └─ exploration.ipynb
├─ src/                    # Code source
│   ├─ data_loader.py      # Chargement et prétraitement des images
│   ├─ model.py            # Définition du modèle IA
│   ├─ trainer.py          # Entraînement et validation
│   └─ predictor.py        # Prédictions sur nouvelles images
├─ web/                    # Interface web (peut être avec Flask ou Streamlit)
│   ├─ app.py              # Point d’entrée de l’app web
│   └─ templates/          # HTML/CSS pour interface
├─ requirements.txt        # Dépendances Python
└─ README.md               # Présentation du projet
```
