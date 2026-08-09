# %% [markdown]
# # Predictive Modeling Optimization - V5 (Direct Stacking, No Classifier Cascade)
# Key insight: classifier mistakes cascade into catastrophic RMSE spikes.
# Solution: let tree-based models learn the zero pattern directly.

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
import sklearn.base
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# %%
df_train = pd.read_csv("train_dataset.csv")
df_test  = pd.read_csv("test_dataset.csv")
print("Train:", df_train.shape, "| Test:", df_test.shape)

# %% [markdown]
# ## 1. Feature Engineering (Physics-Informed)

# %%
def engineer_features(df):
    df = df.copy()
    flow   = df['flow_rate_L_min']
    conc   = df['concentration_mol_L']
    t_in   = df['inlet_temperature_K']
    t_jk   = df['jacket_temperature_K']
    length = df['length_m']

    # --- Core physics ---
    df['residence_time']       = length / flow
    df['temp_delta']           = t_jk - t_in
    df['abs_temp_delta']       = np.abs(t_jk - t_in)
    df['temp_ratio']           = t_jk / t_in
    df['temp_mean']            = (t_jk + t_in) / 2
    df['temp_max']             = np.maximum(t_jk, t_in)
    df['temp_min']             = np.minimum(t_jk, t_in)

    # --- Arrhenius proxies (multiple activation energies) ---
    for Ea in [3000, 5000, 8000, 12000]:
        df[f'arrh_in_{Ea}']    = np.exp(-Ea / t_in)
        df[f'arrh_jk_{Ea}']    = np.exp(-Ea / t_jk)
        df[f'arrh_mean_{Ea}']  = np.exp(-Ea / df['temp_mean'])
        df[f'arrh_delta_{Ea}'] = df[f'arrh_jk_{Ea}'] - df[f'arrh_in_{Ea}']

    # --- Residence-time derived ---
    df['residence_time_sq']    = df['residence_time'] ** 2
    df['residence_time_log']   = np.log1p(df['residence_time'])
    df['residence_time_inv']   = 1.0 / (df['residence_time'] + 1e-6)

    # --- Concentration derived ---
    df['conc_x_residence']     = conc * df['residence_time']
    df['conc_sq']              = conc ** 2
    df['conc_log']             = np.log1p(conc)
    df['conc_x_temp_mean']     = conc * df['temp_mean']

    # --- Flow derived ---
    df['flow_inv']             = 1.0 / flow
    df['flow_log']             = np.log1p(flow)
    df['volume_throughput']    = flow * length
    df['conc_x_flow']          = conc * flow

    # --- Interaction features ---
    df['flow_x_temp_delta']    = flow * df['temp_delta']
    df['length_x_temp_delta']  = length * df['temp_delta']
    df['residence_x_arrh']     = df['residence_time'] * df['arrh_mean_5000']

    # --- Damkohler proxies ---
    df['damkohler_5000']       = df['arrh_mean_5000'] * df['residence_time']
    df['damkohler_8000']       = df['arrh_mean_8000'] * df['residence_time']

    # --- Temperature gradient ---
    df['temp_gradient']        = df['temp_delta'] / (length + 1e-6)
    df['temp_delta_sq']        = df['temp_delta'] ** 2

    # --- Indicator: jacket hotter than inlet ---
    df['jacket_hotter']        = (t_jk > t_in).astype(float)

    # --- Selectivity proxies: ratio of competing Arrhenius rates ---
    df['selectivity_5k_8k']    = df['arrh_mean_5000'] / (df['arrh_mean_8000'] + 1e-12)
    df['selectivity_3k_12k']   = df['arrh_mean_3000'] / (df['arrh_mean_12000'] + 1e-12)

    return df

df_train_feat = engineer_features(df_train.drop(columns=['overall_yield']))
df_train_feat['overall_yield'] = df_train['overall_yield'].values
df_test_feat = engineer_features(df_test)

feature_cols = [c for c in df_train_feat.columns if c != 'overall_yield']
X = df_train_feat[feature_cols].values
y = df_train_feat['overall_yield'].values
X_test_final = df_test_feat[feature_cols].values
feature_names = feature_cols
print(f"{len(feature_cols)} features engineered")

# %% [markdown]
# ## 2. Direct Regression - Tune All Models on Full Data
# No classifier stage. Trees can learn the zero-yield boundary naturally.

# %%
cv5 = KFold(n_splits=5, shuffle=True, random_state=42)

# --- XGBoost (Huber loss for robustness to bimodal distribution) ---
def objective_xgb(trial):
    params = {
        'n_estimators':     trial.suggest_int('n_estimators', 100, 800),
        'max_depth':        trial.suggest_int('max_depth', 2, 8),
        'learning_rate':    trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-5, 50, log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1e-5, 50, log=True),
        'gamma':            trial.suggest_float('gamma', 0, 10),
        'random_state': 42, 'verbosity': 0,
    }
    return cross_val_score(xgb.XGBRegressor(**params), X, y, cv=cv5,
                            scoring='neg_root_mean_squared_error').mean()

print("[1/7] Tuning XGBoost (40 trials)...")
study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=40)
best_xgb_params = study_xgb.best_params
best_xgb_params.update({'random_state': 42, 'verbosity': 0})
print(f"  -> Best XGB RMSE: {-study_xgb.best_value:.4f}")

# --- LightGBM ---
def objective_lgb(trial):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 100, 800),
        'max_depth':         trial.suggest_int('max_depth', 2, 8),
        'learning_rate':     trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 2, 30),
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-5, 50, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-5, 50, log=True),
        'num_leaves':        trial.suggest_int('num_leaves', 8, 63),
        'random_state': 42, 'verbose': -1,
    }
    return cross_val_score(lgb.LGBMRegressor(**params), X, y, cv=cv5,
                            scoring='neg_root_mean_squared_error').mean()

print("[2/7] Tuning LightGBM (40 trials)...")
study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(objective_lgb, n_trials=40)
best_lgb_params = study_lgb.best_params
best_lgb_params.update({'random_state': 42, 'verbose': -1})
print(f"  -> Best LGB RMSE: {-study_lgb.best_value:.4f}")

# --- ExtraTrees ---
def objective_et(trial):
    params = {
        'n_estimators':     trial.suggest_int('n_estimators', 100, 800),
        'max_depth':        trial.suggest_int('max_depth', 3, 25),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'min_samples_split':trial.suggest_int('min_samples_split', 2, 15),
        'max_features':     trial.suggest_float('max_features', 0.2, 1.0),
        'random_state': 42,
    }
    return cross_val_score(ExtraTreesRegressor(**params), X, y, cv=cv5,
                            scoring='neg_root_mean_squared_error').mean()

print("[3/7] Tuning ExtraTrees (40 trials)...")
study_et = optuna.create_study(direction='maximize')
study_et.optimize(objective_et, n_trials=40)
best_et_params = study_et.best_params
best_et_params.update({'random_state': 42})
print(f"  -> Best ET RMSE: {-study_et.best_value:.4f}")

# --- GradientBoosting ---
def objective_gbr(trial):
    params = {
        'n_estimators':     trial.suggest_int('n_estimators', 100, 800),
        'max_depth':        trial.suggest_int('max_depth', 2, 8),
        'learning_rate':    trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 15),
        'min_samples_split':trial.suggest_int('min_samples_split', 2, 15),
        'max_features':     trial.suggest_float('max_features', 0.3, 1.0),
        'loss':             'huber',  # robust to outliers/bimodal distribution
        'random_state': 42,
    }
    return cross_val_score(GradientBoostingRegressor(**params), X, y, cv=cv5,
                            scoring='neg_root_mean_squared_error').mean()

print("[4/7] Tuning GradientBoosting-Huber (40 trials)...")
study_gbr = optuna.create_study(direction='maximize')
study_gbr.optimize(objective_gbr, n_trials=40)
best_gbr_params = study_gbr.best_params
best_gbr_params.update({'random_state': 42, 'loss': 'huber'})
print(f"  -> Best GBR RMSE: {-study_gbr.best_value:.4f}")

# --- RandomForest ---
def objective_rf(trial):
    params = {
        'n_estimators':     trial.suggest_int('n_estimators', 100, 800),
        'max_depth':        trial.suggest_int('max_depth', 3, 25),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'min_samples_split':trial.suggest_int('min_samples_split', 2, 15),
        'max_features':     trial.suggest_float('max_features', 0.2, 1.0),
        'random_state': 42,
    }
    return cross_val_score(RandomForestRegressor(**params), X, y, cv=cv5,
                            scoring='neg_root_mean_squared_error').mean()

print("[5/7] Tuning RandomForest (30 trials)...")
study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(objective_rf, n_trials=30)
best_rf_params = study_rf.best_params
best_rf_params.update({'random_state': 42})
print(f"  -> Best RF RMSE: {-study_rf.best_value:.4f}")

# --- MLPRegressor ---
def objective_mlp(trial):
    params = {
        'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(64, 64), (128, 64), (64, 32, 16)]),
        'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
        'alpha': trial.suggest_float('alpha', 1e-4, 10.0, log=True),
        'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True),
        'max_iter': 500,
        'early_stopping': True,
        'random_state': 42
    }
    model = make_pipeline(StandardScaler(), MLPRegressor(**params))
    return cross_val_score(model, X, y, cv=cv5,
                            scoring='neg_root_mean_squared_error').mean()

print("[6/7] Tuning MLPRegressor (30 trials)...")
study_mlp = optuna.create_study(direction='maximize')
study_mlp.optimize(objective_mlp, n_trials=30)
best_mlp_params = study_mlp.best_params
best_mlp_params.update({'max_iter': 500, 'early_stopping': True, 'random_state': 42})
print(f"  -> Best MLP RMSE: {-study_mlp.best_value:.4f}")

# --- CatBoostRegressor ---
def objective_cb(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 800),
        'depth': trial.suggest_int('depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'random_state': 42,
        'verbose': 0
    }
    return cross_val_score(CatBoostRegressor(**params), X, y, cv=cv5,
                            scoring='neg_root_mean_squared_error').mean()

print("[7/7] Tuning CatBoostRegressor (30 trials)...")
study_cb = optuna.create_study(direction='maximize')
study_cb.optimize(objective_cb, n_trials=30)
best_cb_params = study_cb.best_params
best_cb_params.update({'random_state': 42, 'verbose': 0})
print(f"  -> Best CB RMSE: {-study_cb.best_value:.4f}")

# %% [markdown]
# ## 3. Multi-Seed Stacking (15 base models -> Ridge)
# Each model type x 3 random seeds + single seed RF/GBR/MLP = 15 diverse base learners

# %%
print("\nBuilding multi-seed stacking ensemble...")

seed_list = [42, 123, 7]
models_config = []

for seed in seed_list:
    p = {**best_xgb_params, 'random_state': seed}
    models_config.append((f'XGB_s{seed}', xgb.XGBRegressor(**p)))

    p = {**best_lgb_params, 'random_state': seed}
    models_config.append((f'LGB_s{seed}', lgb.LGBMRegressor(**p)))

    p = {**best_et_params, 'random_state': seed}
    models_config.append((f'ET_s{seed}', ExtraTreesRegressor(**p)))

    p = {**best_cb_params, 'random_state': seed}
    models_config.append((f'CB_s{seed}', CatBoostRegressor(**p)))

# GBR and RF with single seed (already diverse enough)
models_config.append(('GBR', GradientBoostingRegressor(**best_gbr_params)))
models_config.append(('RF', RandomForestRegressor(**best_rf_params)))
models_config.append(('MLP', make_pipeline(StandardScaler(), MLPRegressor(**best_mlp_params))))

print(f"  Total base models: {len(models_config)}")

# Generate OOF predictions for stacking
n_total = X.shape[0]
oof_preds = np.zeros((n_total, len(models_config)))

for mi, (name, template) in enumerate(models_config):
    for tr_idx, vl_idx in cv5.split(X):
        m = sklearn.base.clone(template)
        m.fit(X[tr_idx], y[tr_idx])
        oof_preds[vl_idx, mi] = m.predict(X[vl_idx])
    oof_rmse = np.sqrt(mean_squared_error(y, oof_preds[:, mi]))
    print(f"  {name:12s} OOF RMSE: {oof_rmse:.4f}")

# Clip negative predictions to 0
oof_preds = np.clip(oof_preds, 0, 100)

# Tune Ridge alpha via Optuna
def objective_meta(trial):
    alpha = trial.suggest_float('alpha', 1e-4, 100, log=True)
    meta = Ridge(alpha=alpha)
    scores = cross_val_score(meta, oof_preds, y, cv=cv5,
                              scoring='neg_root_mean_squared_error')
    return scores.mean()

print("\n  Tuning meta-learner alpha (30 trials)...")
study_meta = optuna.create_study(direction='maximize')
study_meta.optimize(objective_meta, n_trials=30)
best_alpha = study_meta.best_params['alpha']
print(f"  -> Best Ridge alpha: {best_alpha:.4f}")

meta = Ridge(alpha=best_alpha)
meta.fit(oof_preds, y)
meta_pred = np.clip(meta.predict(oof_preds), 0, 100)
meta_rmse = np.sqrt(mean_squared_error(y, meta_pred))
print(f"  Stacked OOF RMSE: {meta_rmse:.4f}")
print(f"  Weights: { {n: round(w,3) for n,w in zip([n for n,_ in models_config], meta.coef_)} }")

# Fit all base models on full data
final_models = []
for name, template in models_config:
    m = sklearn.base.clone(template)
    m.fit(X, y)
    final_models.append((name, m))

# %% [markdown]
# ## 4. Plots

# %%
y_va = y
y_pa = meta_pred

plt.figure(figsize=(8, 8))
plt.scatter(y_va, y_pa, alpha=0.6, color='steelblue', edgecolors='white', s=60)
plt.plot([0, 100], [0, 100], 'r--', lw=2)
plt.xlabel('True Yield')
plt.ylabel('Predicted Yield')
plt.title(f'True vs Predicted (CV RMSE={meta_rmse:.2f})')
plt.savefig('true_vs_predicted_yield.png', dpi=300)
plt.close()
print("Saved true_vs_predicted_yield.png")

# Feature importance (from best XGBoost)
xgb_fi = xgb.XGBRegressor(**best_xgb_params)
xgb_fi.fit(X, y)
imp = xgb_fi.feature_importances_
si = np.argsort(imp)[-20:]  # top 20
plt.figure(figsize=(10, 8))
plt.barh(range(len(si)), imp[si], align='center')
plt.yticks(range(len(si)), np.array(feature_names)[si])
plt.xlabel('Feature Importance')
plt.title('XGBoost Top-20 Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.close()
print("Saved feature_importance.png")

# %% [markdown]
# ## 5. Generate Submission

# %%
bp_test = np.column_stack([m.predict(X_test_final) for _, m in final_models])
bp_test = np.clip(bp_test, 0, 100)
pred_test = np.clip(meta.predict(bp_test), 0, 100)

submission = pd.DataFrame({'overall_yield': np.round(pred_test, 3)})
submission.to_csv('Ctrl+Alt+Achieve.csv', index=False)
print("\nSubmission saved as Ctrl+Alt+Achieve.csv!")
print(submission)
print(f"\n{(pred_test < 1).sum()} near-zeros, {(pred_test >= 1).sum()} positives")
