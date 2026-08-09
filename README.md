# Predictive Modeling Optimization Challenge

This repository contains the solution for the **ML Hackathon: The Predictive Modeling Optimization Challenge**. The objective is to build a data-driven machine learning surrogate model to predict the overall yield of a desired product in a continuous flow reactor, replacing computationally expensive physics-based simulations.

## Project Structure

- **`eda.ipynb` / `eda.py`**: Exploratory Data Analysis (EDA).
- **`train.ipynb` / `train.py`**: Model training and evaluation. To tackle the massive bimodal data cliffs, we implemented a custom **V3 Two-Stage Stacking Pipeline**. The code utilizes an Optuna hyperparameter optimization engine, a Ridge Meta-Learner, hard thresholding for bimodal predictions, and 26 advanced thermodynamic and kinetic physics features.
- **`Ctrl+Alt+Achieve.csv`**: The final prediction submission file formatted strictly according to the hackathon guidelines.
- **`pitch_and_evaluation_notes.md`**: A detailed document outlining our V3 approach for Phase 2, covering our Optuna implementation, Ridge Stacking architecture, and hard-thresholding techniques.
- **`train_dataset.csv` & `test_dataset.csv`**: The historical plant data provided for training and the unseen data for generating final predictions.

## How to Run

1. Ensure you have Python 3 installed.
2. Install the required dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter xgboost lightgbm optuna
   ```
3. Run the Jupyter Notebooks to see the step-by-step analysis and model execution:
   ```bash
   jupyter notebook train.ipynb
   ```
   Alternatively, you can run the raw Python script:
   ```bash
   python train.py
   ```

## Model Highlights (V3 Architecture)

- **Algorithm:** Optuna-Optimized Stacking Ensemble (XGBoost + LightGBM + ExtraTrees + GBR -> Ridge Meta-Learner)
- **Validation RMSE (5-Fold CV):** ~18.66
- **Hard Threshold Blending:** Instead of soft blending `Pred * Probability`, which artificially attenuates confident high-yield predictions, we explicitly optimize a hard cutoff threshold (`P(success) > 0.450`).
- **Feature Explosion (26 Vectors):** We augmented the small dataset with advanced physics terms like Damkohler proxies (`reaction_rate * residence_time`), temperature ratios, logarithmic flow rates, and squared "over-cooking" penalties.
