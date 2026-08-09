# %% [markdown]
# # Predictive Modeling Optimization Challenge - Model Training (V3 Stacking & Optuna)
# 
# ## 1. Introduction
# This is our V3 strategy, designed to aggressively target an RMSE of 5-10. 
# 
# **Key Architecture Changes:**
# 1. **26-Feature Physics Expansion**: Damkohler proxies, Log transforms, and Temperature interactions.
# 2. **Optuna Auto-Tuning**: 250 trials to find mathematically optimal hyperparameters and classification thresholds (0.3 - 0.75).
# 3. **Stacking Regressor (Meta-Learner)**: Using `Ridge` regression to learn the perfect ensemble blend from Out-of-Fold predictions instead of relying on arbitrary inverse-RMSE weights.
# 4. **Hard Classification**: Using a sharp cutoff instead of soft-blending `Pred * Probability` to prevent artificial attenuation of high-yield runs.
# 5. **Zero Cutoff = 0.2**: Captures legitimate near-zero reactions.

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, GradientBoostingClassifier, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings and Optuna output for a clean notebook
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# %% [markdown]
# ## 2. Load the Datasets

# %%
df_train = pd.read_csv("train_dataset.csv")
df_test = pd.read_csv("test_dataset.csv")

print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)

# %% [markdown]
# ## 3. Physics-Informed Feature Engineering (26 Features)

# %%
def feature_engineering(df):
    df_engineered = df.copy()
    
    # --- Base Physical Metrics ---
    df_engineered['residence_time'] = df_engineered['length_m'] / df_engineered['flow_rate_L_min']
    df_engineered['temp_delta'] = df_engineered['jacket_temperature_K'] - df_engineered['inlet_temperature_K']
    df_engineered['temp_ratio'] = df_engineered['jacket_temperature_K'] / df_engineered['inlet_temperature_K']
    df_engineered['mean_temp'] = (df_engineered['jacket_temperature_K'] + df_engineered['inlet_temperature_K']) / 2.0
    
    # --- Arrhenius & Thermodynamics ---
    df_engineered['arrhenius_inlet'] = np.exp(-5000.0 / df_engineered['inlet_temperature_K'])
    df_engineered['arrhenius_jacket'] = np.exp(-5000.0 / df_engineered['jacket_temperature_K'])
    df_engineered['arrhenius_mean'] = np.exp(-5000.0 / df_engineered['mean_temp'])
    
    # --- Reaction Kinetics ---
    df_engineered['conc_x_residence'] = df_engineered['concentration_mol_L'] * df_engineered['residence_time']
    df_engineered['volume_throughput'] = df_engineered['length_m'] * df_engineered['flow_rate_L_min']
    
    # Damkohler Proxy (Reaction Rate vs Mass Transfer)
    df_engineered['damkohler_proxy'] = df_engineered['arrhenius_jacket'] * df_engineered['residence_time']
    
    # --- Non-linear Polynomials (Over-cooking / Extreme limits) ---
    df_engineered['residence_time_sq'] = df_engineered['residence_time'] ** 2
    df_engineered['temp_delta_sq'] = df_engineered['temp_delta'] ** 2
    df_engineered['flow_rate_sq'] = df_engineered['flow_rate_L_min'] ** 2
    
    # --- Logarithmic Transforms (Diminishing returns) ---
    df_engineered['log_flow_rate'] = np.log1p(df_engineered['flow_rate_L_min'])
    df_engineered['log_residence_time'] = np.log1p(df_engineered['residence_time'])
    
    # --- Advanced Interactions ---
    df_engineered['temp_flow_interaction'] = df_engineered['mean_temp'] * df_engineered['flow_rate_L_min']
    df_engineered['conc_temp_interaction'] = df_engineered['concentration_mol_L'] * df_engineered['mean_temp']
    
    return df_engineered

X_raw = df_train.drop(columns=['overall_yield'])
y = df_train['overall_yield']
X_test_raw = df_test.copy()

X = feature_engineering(X_raw)
X_test_final = feature_engineering(X_test_raw)

print("Engineered Feature Count:", X.shape[1])

# %% [markdown]
# ## 4. Optuna Auto-Tuning & Meta-Learner Stacking
# We will run 250 Optuna trials to globally optimize:
# 1. Classification Threshold (0.3 to 0.75).
# 2. Hyperparameters for `XGBoost`, `LightGBM`, `ExtraTrees`, and `GradientBoosting`.
# 3. The `Ridge` Meta-Learner Alpha.

# %%
def objective(trial):
    # Suggest hyperparameters
    class_threshold = trial.suggest_float('class_threshold', 0.3, 0.75)
    
    xgb_depth = trial.suggest_int('xgb_depth', 2, 5)
    xgb_lr = trial.suggest_float('xgb_lr', 0.01, 0.1, log=True)
    
    lgb_depth = trial.suggest_int('lgb_depth', 2, 5)
    lgb_lr = trial.suggest_float('lgb_lr', 0.01, 0.1, log=True)
    
    rf_depth = trial.suggest_int('rf_depth', 3, 6)
    gbr_depth = trial.suggest_int('gbr_depth', 2, 5)
    
    ridge_alpha = trial.suggest_float('ridge_alpha', 0.1, 10.0, log=True)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    
    for train_idx, val_idx in kf.split(X):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # Stage 1: Classifier (Target = y > 0.2 to capture near-zeros)
        y_train_class = (y_train > 0.2).astype(int)
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        clf.fit(X_train, y_train_class)
        
        # Identify successful runs in training fold
        mask = y_train > 0.2
        X_train_sub = X_train[mask]
        y_train_sub = y_train[mask]
        
        # Define Base Regressors
        regressors = [
            XGBRegressor(n_estimators=150, max_depth=xgb_depth, learning_rate=xgb_lr, random_state=42, eval_metric='rmse'),
            LGBMRegressor(n_estimators=150, max_depth=lgb_depth, learning_rate=lgb_lr, random_state=42, verbose=-1),
            ExtraTreesRegressor(n_estimators=150, max_depth=rf_depth, random_state=42),
            GradientBoostingRegressor(n_estimators=150, max_depth=gbr_depth, random_state=42)
        ]
        
        # OOF Meta-features for Ridge training
        meta_X_train = np.zeros((len(X_train_sub), len(regressors)))
        
        # Inner CV to generate meta-features for Ridge training safely
        inner_kf = KFold(n_splits=3, shuffle=True, random_state=42)
        for inner_train_idx, inner_val_idx in inner_kf.split(X_train_sub):
            inner_X_train, inner_y_train = X_train_sub.iloc[inner_train_idx], y_train_sub.iloc[inner_train_idx]
            inner_X_val = X_train_sub.iloc[inner_val_idx]
            
            for idx, reg in enumerate(regressors):
                reg.fit(inner_X_train, inner_y_train)
                meta_X_train[inner_val_idx, idx] = np.clip(reg.predict(inner_X_val), 0.0, 100.0)
                
        # Train Meta-Learner (Ridge) on the OOF meta-features
        meta_learner = Ridge(alpha=ridge_alpha, positive=True) # positive=True prevents negative weights
        meta_learner.fit(meta_X_train, y_train_sub)
        
        # Retrain Base Regressors on FULL subset
        for reg in regressors:
            reg.fit(X_train_sub, y_train_sub)
            
        # Predict on Validation Fold
        meta_X_val = np.zeros((len(X_val), len(regressors)))
        for idx, reg in enumerate(regressors):
            meta_X_val[:, idx] = np.clip(reg.predict(X_val), 0.0, 100.0)
            
        # Meta-prediction
        val_ens_preds = np.clip(meta_learner.predict(meta_X_val), 0.0, 100.0)
        
        # Stage 2: Hard Thresholding Blending
        p_success_val = clf.predict_proba(X_val)[:, 1]
        
        # If P(success) > optimized threshold, we use the regressor. Otherwise, 0.
        final_val_preds = np.where(p_success_val > class_threshold, val_ens_preds, 0.0)
        oof_preds[val_idx] = final_val_preds

    return np.sqrt(mean_squared_error(y, oof_preds))

print("Starting Optuna optimization (50 trials)...")
optuna.logging.set_verbosity(optuna.logging.INFO)
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50, n_jobs=-1)

print(f"\\nBest Trial RMSE: {study.best_value:.4f}")
print("Best Hyperparameters:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# %% [markdown]
# ## 5. Train Final Optimal Stacking Model
# Now that Optuna found the mathematical optimum, we train the absolute final model across all 5 folds to generate our Test Predictions.

# %%
best_params = study.best_params
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds_final = np.zeros(len(X))
final_test_preds = np.zeros(len(X_test_final))

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # Classifier
    y_train_class = (y_train > 0.2).astype(int)
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    clf.fit(X_train, y_train_class)
    
    # Subsetting
    mask = y_train > 0.2
    X_train_sub, y_train_sub = X_train[mask], y_train[mask]
    
    regressors = [
        XGBRegressor(n_estimators=150, max_depth=best_params['xgb_depth'], learning_rate=best_params['xgb_lr'], random_state=42, eval_metric='rmse'),
        LGBMRegressor(n_estimators=150, max_depth=best_params['lgb_depth'], learning_rate=best_params['lgb_lr'], random_state=42, verbose=-1),
        ExtraTreesRegressor(n_estimators=150, max_depth=best_params['rf_depth'], random_state=42),
        GradientBoostingRegressor(n_estimators=150, max_depth=best_params['gbr_depth'], random_state=42)
    ]
    
    meta_X_train = np.zeros((len(X_train_sub), len(regressors)))
    inner_kf = KFold(n_splits=3, shuffle=True, random_state=42)
    for inner_train_idx, inner_val_idx in inner_kf.split(X_train_sub):
        inner_X_train, inner_y_train = X_train_sub.iloc[inner_train_idx], y_train_sub.iloc[inner_train_idx]
        inner_X_val = X_train_sub.iloc[inner_val_idx]
        
        for idx, reg in enumerate(regressors):
            reg.fit(inner_X_train, inner_y_train)
            meta_X_train[inner_val_idx, idx] = np.clip(reg.predict(inner_X_val), 0.0, 100.0)
            
    meta_learner = Ridge(alpha=best_params['ridge_alpha'], positive=True)
    meta_learner.fit(meta_X_train, y_train_sub)
    
    # Train full fold
    for reg in regressors:
        reg.fit(X_train_sub, y_train_sub)
        
    # --- Prediction on Valid ---
    meta_X_val = np.zeros((len(X_val), len(regressors)))
    for idx, reg in enumerate(regressors):
        meta_X_val[:, idx] = np.clip(reg.predict(X_val), 0.0, 100.0)
        
    val_ens_preds = np.clip(meta_learner.predict(meta_X_val), 0.0, 100.0)
    p_success_val = clf.predict_proba(X_val)[:, 1]
    oof_preds_final[val_idx] = np.where(p_success_val > best_params['class_threshold'], val_ens_preds, 0.0)
    
    # --- Prediction on Test ---
    meta_X_test = np.zeros((len(X_test_final), len(regressors)))
    for idx, reg in enumerate(regressors):
        meta_X_test[:, idx] = np.clip(reg.predict(X_test_final), 0.0, 100.0)
        
    test_ens_preds = np.clip(meta_learner.predict(meta_X_test), 0.0, 100.0)
    p_success_test = clf.predict_proba(X_test_final)[:, 1]
    
    final_test_preds += np.where(p_success_test > best_params['class_threshold'], test_ens_preds, 0.0) / 5.0

print(f"Final Sanity Check (V3 Stacking RMSE): {np.sqrt(mean_squared_error(y, oof_preds_final)):.4f}")

# %% [markdown]
# ## 6. Visualizations

# %%
plt.figure(figsize=(8, 8))
plt.scatter(y, oof_preds_final, alpha=0.7, color='green')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('True Yield')
plt.ylabel('Predicted Yield (V3 Stacking)')
plt.title('True vs Predicted Yield (Optuna Optimized Stacking)')
plt.savefig('true_vs_predicted_yield.png', dpi=300)
plt.close()

# %% [markdown]
# ## 7. Prepare Submission File

# %%
submission = pd.DataFrame({'overall_yield': np.round(final_test_preds, 3)})
submission.to_csv('Ctrl+Alt+Achieve.csv', index=False)

print("Submission saved successfully as Ctrl+Alt+Achieve.csv!")
print(submission.head())
