# Predictive Modeling Optimization Challenge

This repository contains the solution for the **ML Hackathon: The Predictive Modeling Optimization Challenge**. The objective is to build a data-driven machine learning surrogate model to predict the overall yield of a desired product in a continuous flow reactor, replacing computationally expensive physics-based simulations.

## Project Structure

- **`eda.ipynb` / `eda.py`**: Exploratory Data Analysis (EDA). This step includes loading the data, visualizing distributions, analyzing bivariate relationships, and generating a correlation matrix to understand the underlying thermodynamic and kinetic behaviors.
- **`train.ipynb` / `train.py`**: Model training and evaluation. We adopted a Physics-Informed ML approach using **Extra Trees, Gaussian Process, and SVR** regressors to predict the target (`overall_yield`). The code includes physics-based feature engineering (Arrhenius terms, residence time), model training, evaluation (RMSE), feature importance extraction, bounding predictions to [0, 100], and generating final predictions on the unseen test dataset.
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

- **Algorithm:** Physics-Informed Ensemble (Extra Trees + Gaussian Process + SVR)
- **Validation RMSE:** ~20.12
- **Why Smooth Regressors?** Standard tree boosting creates step-functions that poorly mimic chemical kinetics. By introducing explicit physics terms (like $e^{-C/T}$) and relying on models like Gaussian Processes and SVR, we smoothly map the bounded continuous output surface without breaking physical laws (strictly clipping between 0% and 100% yield).
