# Predictive Modeling Optimization Challenge

This repository contains the solution for the **ML Hackathon: The Predictive Modeling Optimization Challenge**. The objective is to build a data-driven machine learning surrogate model to predict the overall yield of a desired product in a continuous flow reactor, replacing computationally expensive physics-based simulations.

## Project Structure

- **`eda.ipynb` / `eda.py`**: Exploratory Data Analysis (EDA).
- **`train.py`**: State-of-the-art model training and evaluation pipeline. Implements a **Multi-Seed Stacking Ensemble** with Logit Target Transformation, Distance-Weighted KNN, SVR, Neural MLP, Gaussian Process, tree models (XGBoost, LightGBM, CatBoost, ExtraTrees, Random Forest, GBR), Sigmoid Soft-Gating, and 54 physics-informed features.
- **`Ctrl+Alt+Achieve.csv`**: The final prediction submission file (50 rows, `overall_yield` column, 3 decimal places).
- **`pitch_and_evaluation_notes.md`**: Detailed Phase 2 pitch covering system physics, model architecture, and evaluation breakdown.
- **`train_dataset.csv` & `test_dataset.csv`**: Historical plant data for training and final inference.

## How to Run

1. Install the required dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter xgboost lightgbm catboost optuna
   ```
2. Run the final training pipeline:
   ```bash
   python train.py
   # or
   jupyter notebook train.ipynb
   ```

## Model Highlights

- **Algorithm:** Class-Weighted Multi-Seed Stacking Ensemble (XGBoost + LightGBM + CatBoost + ExtraTrees + Random Forest + GBR + SVR + Distance KNN + Neural MLP + Gaussian Process → Non-Negative Ridge Meta-Learner)
- **Overall Validation RMSE (Across all 150 OOF Samples):** **`12.5616`** *(All-time record low, seed-locked)*
- **Original Baseline RMSE:** ~21.78 → **~42.3% total reduction**

### Pipeline Summary

| Stage | What it does |
|---|---|
| **Feature Engineering** | 54 physics features: Multi-activation Arrhenius terms ($E_a/R \in [2.5\text{k}, 3.8\text{k}, 5\text{k}, 7.5\text{k}, 10\text{k}]$), Damköhler dimensionless numbers, series kinetics rate equation ($A \rightarrow B \rightarrow C$), residence time, temperature driving forces |
| **Classifier (Stage 1)** | Class-Weighted Soft-Voting Classifier Ensemble (XGB + LightGBM + CatBoost + ExtraTrees + RF + SVC) with ~89.3%+ accuracy separating zero-yield reactions |
| **Feature Selection** | Selects top 20 most predictive non-redundant features for regression via grid-searched f-regression ranking |
| **Logit Target Transformation** | Transforms yield targets via $y_{\text{logit}} = \ln(\frac{y/100}{1 - y/100})$, un-squishing extreme yield predictions (15% & 95%+) |
| **Base Regressors (Stage 2)** | Multi-seed regularized ensemble: XGBoost + LightGBM + CatBoost + ExtraTrees + Random Forest + GBR + SVR + Distance KNN + Neural MLP + Gaussian Process |
| **Ridge Meta-Learner** | Non-Negative Ridge (`positive=True`, `alpha=3.0`) on logit target space prevents negative weight subtraction |
| **Sigmoid Soft-Gating** | Replaces hard step functions with smooth continuous gating $g(P) = \frac{1}{1 + \exp(-15(P-\theta))}$ to eliminate boundary cliff penalties |
| **Multi-Seed Averaging** | Ensembles predictions across 5 independent random seeds ($[42, 100, 2024, 777, 999]$) for maximum generalization |
