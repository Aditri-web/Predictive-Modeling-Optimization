# Predictive Modeling Optimization Challenge

This repository contains the solution for the **ML Hackathon: The Predictive Modeling Optimization Challenge**. The objective is to build a data-driven machine learning surrogate model to predict the overall yield of a desired product in a continuous flow reactor, replacing computationally expensive physics-based simulations.

## Project Structure

- **`eda.ipynb` / `eda.py`**: Exploratory Data Analysis (EDA).
- **`train.ipynb` / `train.py`**: Model training and evaluation. Implements a **V5 Multi-Seed Stacking Pipeline** with Optuna hyperparameter optimization, Ridge Meta-Learner, CatBoost symmetric tree integration, and 25 advanced thermodynamic/kinetic physics features.
- **`Ctrl+Alt+Achieve.csv`**: The final prediction submission file (50 rows, `overall_yield` column, 3 decimal places).
- **`pitch_and_evaluation_notes.md`**: Detailed Phase 2 pitch covering the full V5 architecture.
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

- **Algorithm:** Multi-Seed Stacking Ensemble (XGBoost + LightGBM + ExtraTrees + CatBoost + GBR + RF + MLP → Ridge Meta-Learner)
- **Validation RMSE (Locked Stacked OOF):** **10.09**
- **Previous V3 RMSE:** ~14.64 → **~31% improvement**
- **Original Baseline RMSE:** ~21.78 → **~53% total reduction**

### Pipeline Summary

| Stage | What it does |
|---|---|
| **Feature Engineering** | 25 physics features: Arrhenius terms, Damkohler proxy, residence time, temperature interactions, log/poly transforms |
| **Multi-Seed Base Learners** | Optuna-tuned `XGBoost`, `LightGBM`, `ExtraTrees`, and `CatBoost` run across 3 random seeds each, plus single-seed GBR, RF, and MLP (15 total base models) |
| **Ridge Meta-Learner** | Tuned Ridge regression model learns the optimal blend of all 15 base models using strictly out-of-fold predictions |
| **CatBoost Integration** | Symmetric tree gradient boosting inherently resists overfitting on the tiny 150-row dataset, pulling the ensemble error heavily down |
