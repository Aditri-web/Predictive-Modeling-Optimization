# %% [markdown]
# # Predictive Modeling Optimization Challenge - Model Training
# 
# ## 1. Introduction
# In this notebook, we build a robust predictive model to predict the `overall_yield` of Product B. Following physical constraints and continuous process behavior, we utilize physics-informed feature engineering and smooth regressors (Extra Trees, Gaussian Processes, SVR).

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor, VotingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# %% [markdown]
# ## 2. Load the Datasets

# %%
df_train = pd.read_csv("train_dataset.csv")
df_test = pd.read_csv("test_dataset.csv")

print(df_train.head())
print(df_test.head())

# %% [markdown]
# ## 3. Physics-Informed Feature Engineering
# We will define a function to create chemistry-based features to accurately reflect the underlying physical laws.

# %%
def feature_engineering(df):
    df_engineered = df.copy()
    
    # 1. Residence Time Proxy (tau) = length / flow_rate
    df_engineered['residence_time_tau'] = df_engineered['length_m'] / df_engineered['flow_rate_L_min']
    
    # 2. Arrhenius Terms = exp(-Constant / T)
    # Using a scaling factor (e.g., 1000) so the exp() values don't underflow to 0
    df_engineered['arrhenius_jacket'] = np.exp(-1000.0 / df_engineered['jacket_temperature_K'])
    df_engineered['arrhenius_inlet'] = np.exp(-1000.0 / df_engineered['inlet_temperature_K'])
    
    # 3. Heat Input Proxy = delta_T * tau
    df_engineered['heat_input_proxy'] = (df_engineered['jacket_temperature_K'] - df_engineered['inlet_temperature_K']) * df_engineered['residence_time_tau']
    
    # 4. Initial Reactant Mass Rate = concentration * flow_rate
    df_engineered['initial_mass_rate'] = df_engineered['concentration_mol_L'] * df_engineered['flow_rate_L_min']
    
    return df_engineered

X_train_raw = df_train.drop(columns=['overall_yield'])
y = df_train['overall_yield']
X_test_raw = df_test.copy()

X = feature_engineering(X_train_raw)
X_test_final = feature_engineering(X_test_raw)

# Split for local validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
print(f"Val shapes: X={X_val.shape}, y={y_val.shape}")

# %% [markdown]
# ## 4. Smooth Ensemble Modeling
# We will combine Extra Trees, Gaussian Process Regressor, and SVR to create a robust Voting Regressor. These models map continuous surfaces better than standard step-function boosting models.
# All inputs will be StandardScaled, as GPR and SVR require normalized inputs to function properly.

# %%
# Define individual models
et_model = Pipeline([
    ('scaler', StandardScaler()),
    ('et', ExtraTreesRegressor(n_estimators=300, max_depth=8, random_state=42, min_samples_split=4))
])

# Gaussian Process with a smooth Matern kernel + noise
kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1)
gp_model = Pipeline([
    ('scaler', StandardScaler()),
    ('gp', GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42, normalize_y=True))
])

svr_model = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(kernel='rbf', C=50, epsilon=0.1, gamma='scale'))
])

# Create an Ensemble Model
ensemble_model = VotingRegressor([
    ('et', et_model),
    ('gp', gp_model),
    ('svr', svr_model)
])

# Cross-validation score
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Custom evaluation that strictly bounds predictions to [0, 100]
def bounded_rmse_cv(model, X_full, y_full, kfold):
    rmses = []
    for train_idx, test_idx in kfold.split(X_full):
        X_tr, X_te = X_full.iloc[train_idx], X_full.iloc[test_idx]
        y_tr, y_te = y_full.iloc[train_idx], y_full.iloc[test_idx]
        
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        
        # Bounding transformation
        preds_clipped = np.clip(preds, 0.0, 100.0)
        rmses.append(np.sqrt(mean_squared_error(y_te, preds_clipped)))
    return np.mean(rmses)

cv_rmse = bounded_rmse_cv(ensemble_model, X, y, kf)

print(f"Cross-Validation RMSE (Bounded Ensemble): {cv_rmse:.4f}")

# Fit on training fold and validate
ensemble_model.fit(X_train, y_train)
y_val_pred_raw = ensemble_model.predict(X_val)
y_val_pred = np.clip(y_val_pred_raw, 0.0, 100.0)

val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"Hold-out Validation RMSE: {val_rmse:.4f}")

# %% [markdown]
# ## 5. Model Evaluation & Visualization

# %%
plt.figure(figsize=(8, 8))
plt.scatter(y_val, y_val_pred, alpha=0.7, color='b')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('True Yield')
plt.ylabel('Predicted Yield (Bounded)')
plt.title('True vs Predicted Yield (Physics-Informed Ensemble)')
plt.savefig('true_vs_predicted_yield.png', dpi=300)
plt.close()

# %% [markdown]
# ### Feature Importance (From Extra Trees)

# %%
# Extract feature importances from the ExtraTrees model in the pipeline
et_model.fit(X, y)
feature_importance = et_model.named_steps['et'].feature_importances_
sorted_idx = np.argsort(feature_importance)

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(X.columns)[sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Extra Trees - Feature Importance')
plt.savefig('feature_importance.png', dpi=300)
plt.close()

# %% [markdown]
# ## 6. Retrain on Full Data and Predict for Test Set

# %%
# Fit the ensemble model on ALL available training data
ensemble_model.fit(X, y)

# Predict on test set
final_predictions_raw = ensemble_model.predict(X_test_final)

# Clip final predictions to physical bounds [0, 100]
final_predictions = np.clip(final_predictions_raw, 0.0, 100.0)

# %% [markdown]
# ## 7. Prepare Submission File
# Generate the submission CSV containing exactly 50 rows and one column `overall_yield`, rounded to 3 decimal places.

# %%
submission = pd.DataFrame({'overall_yield': np.round(final_predictions, 3)})
submission.to_csv('Ctrl+Alt+Achieve.csv', index=False)

print("Submission saved successfully as Ctrl+Alt+Achieve.csv!")
print(submission.head())
