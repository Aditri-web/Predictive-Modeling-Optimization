# Predictive Modeling Optimization Challenge

This repository contains the solution for the **ML Hackathon: The Predictive Modeling Optimization Challenge**. The objective is to build a data-driven machine learning surrogate model to predict the overall yield of a desired product in a continuous flow reactor, replacing computationally expensive physics-based simulations.

## Project Structure

- **`eda.ipynb` / `eda.py`**: Exploratory Data Analysis (EDA). This step includes loading the data, visualizing distributions, analyzing bivariate relationships, and generating a correlation matrix to understand the underlying thermodynamic and kinetic behaviors.
- **`train.ipynb` / `train.py`**: Model training and evaluation. To tackle the massive bimodal data cliffs (reaction failure ~0% vs reaction success 70-100%), we implemented a custom **Two-Stage Pipeline** with 5-Fold Cross-Validation. The code includes a custom gradient boosting classifier + regressor cascade, heavily augmented by 7 robust physics-informed features.
- **`Ctrl+Alt+Achieve.csv`**: The final prediction submission file formatted strictly according to the hackathon guidelines (50 rows, 1 column: `overall_yield`, rounded to 3 decimal places).
- **`pitch_and_evaluation_notes.md`**: A detailed document outlining our approach for Phase 2, covering our quantitative validation (RMSE), process insights (physics features), feature engineering innovations, and strategies for model robustness and scalability.
- **`train_dataset.csv` & `test_dataset.csv`**: The historical plant data provided for training and the unseen data for generating final predictions.
- **`*.png`**: Various plots generated during EDA and model evaluation, such as feature importance and correlation matrices.

## How to Run

1. Ensure you have Python 3 installed.
2. Install the required dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```
3. Run the Jupyter Notebooks to see the step-by-step analysis and model execution:
   ```bash
   jupyter notebook eda.ipynb
   jupyter notebook train.ipynb
   ```
   Alternatively, you can run the raw Python scripts:
   ```bash
   python eda.py
   python train.py
   ```

## Model Highlights

- **Algorithm:** Two-Stage Classifier-Regressor Cascade & Inverse-RMSE Weighted Ensemble
- **Validation RMSE (5-Fold CV):** ~18.34
- **Why a Two-Stage Pipeline?** The dataset presents a deep bimodal distribution. Standard regressors shoot right through the middle, generating huge errors on the edges. We built a `GradientBoostingClassifier` to predict reaction success probability, and multiplied this probability by our regressor blend outputs (which were trained *only* on the successful reactions space). This mathematically smooths the error cliff.
- **Physics Injectors:** We augmented the small dataset (150 rows) with 7 custom physics vectors, including Arrhenius `exp(-C/T)` exponential variables, specific residence times ($\tau$), and a squared "over-cooking" penalty metric.
