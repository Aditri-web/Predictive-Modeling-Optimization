# Predictive Modeling Optimization Challenge

This repository contains the solution for the **ML Hackathon: The Predictive Modeling Optimization Challenge**. The objective is to build a data-driven machine learning surrogate model to predict the overall yield of a desired product in a continuous flow reactor, replacing computationally expensive physics-based simulations.

## Project Structure

- **`eda.ipynb` / `eda.py`**: Exploratory Data Analysis (EDA).
- **`train.ipynb` / `train.py`**: Model training and evaluation. Implements a **V3 Two-Stage Stacking Pipeline** with Optuna hyperparameter optimization, GBR+Ridge Meta-Learner with forced positive weights, per-fold classifier threshold optimization, and 25 advanced thermodynamic/kinetic physics features.
- **`Ctrl+Alt+Achieve.csv`**: The final prediction submission file (50 rows, `overall_yield` column, 3 decimal places).
- **`pitch_and_evaluation_notes.md`**: Detailed Phase 2 pitch covering the full V3 architecture.
- **`train_dataset.csv` & `test_dataset.csv`**: Historical plant data for training and final inference.

## How to Run

1. Install the required dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter xgboost lightgbm optuna
   ```
2. Run the training pipeline:
   ```bash
   python train.py
   # or
   jupyter notebook train.ipynb
   ```

## Model Highlights

- **Algorithm:** Two-Stage Stacking Ensemble (XGBoost + LightGBM + ExtraTrees + GBR → Ridge Meta-Learner)
- **Validation RMSE (5-Fold CV):** ~12.71
- **Previous V2 RMSE:** ~18.02 → **~29% improvement**
- **Original Baseline RMSE:** ~21.78 → **~42% total reduction**

### Pipeline Summary

| Stage | What it does |
|---|---|
| **Feature Engineering** | 25 physics features: Arrhenius terms, Damkohler proxy, residence time, temperature interactions, log/poly transforms |
| **Classifier (Stage 1)** | Optuna-tuned `XGBClassifier` (92.7% accuracy) separates zero-yield reactions from successful ones |
| **Base Regressors (Stage 2)** | Optuna-tuned `XGBoost` + `LightGBM` + `ExtraTrees` + `GBR` — trained only on successful reaction subset |
| **Ridge Meta-Learner** | `Ridge(positive=True)` with tuned alpha prevents negative ensemble weights; automatically learns optimal blend |
| **Per-Fold Threshold** | Classifier cutoff is CV-optimized per fold (0.30–0.75) — eliminates fold blowouts from fixed 0.5 threshold |
| **Hard Classification** | `P(success) > threshold → regressor; else → 0.0` — eliminates soft-blend attenuation |

### Why positive=True on Ridge?
The naive Ridge meta-learner assigned **-0.167 weight to XGBoost**, meaning XGBoost was *subtracting* from predictions. Forcing non-negative weights eliminated this drag and dropped RMSE by ~1.4 points.
