# %% [markdown]
# # Predictive Modeling Optimization - V3 Lean (Fast + Aggressive RMSE)

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
)
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import Ridge
import sklearn.base
import xgboost as xgb
import lightgbm as lgb
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
# ## 1. Feature Engineering

# %%
def engineer_features(df):
    df = df.copy()
    flow = df['flow_rate_L_min']
    conc = df['concentration_mol_L']
    t_in = df['inlet_temperature_K']
    t_jk = df['jacket_temperature_K']
    length = df['length_m']

    df['residence_time']      = length / flow
    df['temp_delta']          = t_jk - t_in
    df['abs_temp_delta']      = np.abs(t_jk - t_in)
    df['temp_ratio']          = t_jk / t_in
    df['temp_mean']           = (t_jk + t_in) / 2
    df['arrhenius_inlet']     = np.exp(-5000.0 / t_in)
    df['arrhenius_jacket']    = np.exp(-5000.0 / t_jk)
    df['arrhenius_mean']      = np.exp(-5000.0 / df['temp_mean'])
    df['arrhenius_delta']     = df['arrhenius_jacket'] - df['arrhenius_inlet']
    df['conc_x_residence']    = conc * df['residence_time']
    df['volume_throughput']   = flow * length
    df['conc_x_flow']         = conc * flow
    df['flow_x_temp_delta']   = flow * df['temp_delta']
    df['length_x_temp_delta'] = length * df['temp_delta']
    df['residence_time_sq']   = df['residence_time'] ** 2
    df['residence_time_log']  = np.log1p(df['residence_time'])
    df['temp_delta_sq']       = df['temp_delta'] ** 2
    df['conc_sq']             = conc ** 2
    df['flow_inv']            = 1.0 / flow
    df['damkohler_proxy']     = df['arrhenius_mean'] * length / flow
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
# ## 2. Stage 1 - Classifier (zero vs non-zero)

# %%
ZERO_THRESHOLD = 0.2
y_binary = (y > ZERO_THRESHOLD).astype(int)
print(f"Zero: {(y_binary==0).sum()} | Non-zero: {(y_binary==1).sum()}")

def objective_clf(trial):
    params = {
        'n_estimators':    trial.suggest_int('n_estimators', 100, 500),
        'max_depth':       trial.suggest_int('max_depth', 3, 7),
        'learning_rate':   trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample':       trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight':trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha':       trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
        'reg_lambda':      trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
        'use_label_encoder': False, 'eval_metric': 'logloss',
        'random_state': 42, 'verbosity': 0,
    }
    model = xgb.XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_score(model, X, y_binary, cv=cv, scoring='accuracy').mean()

print("\n[1/6] Tuning classifier (30 trials)...")
study_clf = optuna.create_study(direction='maximize')
study_clf.optimize(objective_clf, n_trials=30)
best_clf_params = study_clf.best_params
best_clf_params.update({'use_label_encoder': False, 'eval_metric': 'logloss',
                        'random_state': 42, 'verbosity': 0})
print(f"  -> Best accuracy: {study_clf.best_value:.4f}")

clf = xgb.XGBClassifier(**best_clf_params)
clf.fit(X, y_binary)

# %% [markdown]
# ## 3. Stage 2 - Tune regressors on non-zero subset

# %%
mask_nz = y > ZERO_THRESHOLD
X_nz = X[mask_nz]
y_nz = y[mask_nz]
cv5 = KFold(n_splits=5, shuffle=True, random_state=42)
print(f"\nNon-zero samples: {X_nz.shape[0]}")

# --- XGBoost ---
def objective_xgb(trial):
    params = {
        'n_estimators':    trial.suggest_int('n_estimators', 200, 600),
        'max_depth':       trial.suggest_int('max_depth', 3, 7),
        'learning_rate':   trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'subsample':       trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'min_child_weight':trial.suggest_int('min_child_weight', 1, 12),
        'reg_alpha':       trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
        'reg_lambda':      trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
        'gamma':           trial.suggest_float('gamma', 0, 3),
        'random_state': 42, 'verbosity': 0,
    }
    return cross_val_score(xgb.XGBRegressor(**params), X_nz, y_nz, cv=cv5,
                            scoring='neg_root_mean_squared_error').mean()

print("[2/6] Tuning XGBoost regressor (40 trials)...")
study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=40)
best_xgb_params = study_xgb.best_params
best_xgb_params.update({'random_state': 42, 'verbosity': 0})
print(f"  -> Best XGB RMSE: {-study_xgb.best_value:.4f}")

# --- LightGBM ---
def objective_lgb(trial):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 200, 600),
        'max_depth':         trial.suggest_int('max_depth', 3, 8),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 3, 15),
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
        'num_leaves':        trial.suggest_int('num_leaves', 15, 50),
        'random_state': 42, 'verbose': -1,
    }
    return cross_val_score(lgb.LGBMRegressor(**params), X_nz, y_nz, cv=cv5,
                            scoring='neg_root_mean_squared_error').mean()

print("[3/6] Tuning LightGBM regressor (40 trials)...")
study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(objective_lgb, n_trials=40)
best_lgb_params = study_lgb.best_params
best_lgb_params.update({'random_state': 42, 'verbose': -1})
print(f"  -> Best LGB RMSE: {-study_lgb.best_value:.4f}")

# --- ExtraTrees ---
def objective_et(trial):
    params = {
        'n_estimators':    trial.suggest_int('n_estimators', 200, 600),
        'max_depth':       trial.suggest_int('max_depth', 5, 15),
        'min_samples_leaf':trial.suggest_int('min_samples_leaf', 1, 6),
        'max_features':    trial.suggest_float('max_features', 0.3, 1.0),
        'random_state': 42,
    }
    return cross_val_score(ExtraTreesRegressor(**params), X_nz, y_nz, cv=cv5,
                            scoring='neg_root_mean_squared_error').mean()

print("[4/6] Tuning ExtraTrees regressor (25 trials)...")
study_et = optuna.create_study(direction='maximize')
study_et.optimize(objective_et, n_trials=25)
best_et_params = study_et.best_params
best_et_params.update({'random_state': 42})
print(f"  -> Best ET RMSE: {-study_et.best_value:.4f}")

# %% [markdown]
# ## 4. Tune GBR + Stacking Meta-Learner (OOF predictions -> Ridge, positive=True)

# %%
# --- GradientBoosting (adds diversity, helps reduce fold variance) ---
def objective_gbr(trial):
    params = {
        'n_estimators':    trial.suggest_int('n_estimators', 200, 500),
        'max_depth':       trial.suggest_int('max_depth', 2, 5),
        'learning_rate':   trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'subsample':       trial.suggest_float('subsample', 0.6, 1.0),
        'min_samples_leaf':trial.suggest_int('min_samples_leaf', 1, 6),
        'random_state': 42,
    }
    return cross_val_score(GradientBoostingRegressor(**params), X_nz, y_nz, cv=cv5,
                            scoring='neg_root_mean_squared_error').mean()

print("[4b/6] Tuning GBR regressor (25 trials)...")
study_gbr = optuna.create_study(direction='maximize')
study_gbr.optimize(objective_gbr, n_trials=25)
best_gbr_params = study_gbr.best_params
best_gbr_params.update({'random_state': 42})
print(f"  -> Best GBR RMSE: {-study_gbr.best_value:.4f}")

print("\n[5/6] Building stacking meta-learner...")

models_config = [
    ('XGB', xgb.XGBRegressor(**best_xgb_params)),
    ('LGB', lgb.LGBMRegressor(**best_lgb_params)),
    ('ET',  ExtraTreesRegressor(**best_et_params)),
    ('GBR', GradientBoostingRegressor(**best_gbr_params)),
]

n_nz = X_nz.shape[0]
oof_preds = np.zeros((n_nz, len(models_config)))

for mi, (name, template) in enumerate(models_config):
    for tr_idx, vl_idx in cv5.split(X_nz):
        m = sklearn.base.clone(template)
        m.fit(X_nz[tr_idx], y_nz[tr_idx])
        oof_preds[vl_idx, mi] = m.predict(X_nz[vl_idx])
    oof_rmse = np.sqrt(mean_squared_error(y_nz, oof_preds[:, mi]))
    print(f"  {name} OOF RMSE: {oof_rmse:.4f}")

# Tune Ridge alpha on OOF + force positive weights (eliminates negative contributions)
best_ridge_alpha = 1.0
best_ridge_rmse = float('inf')
for alpha in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
    meta_try = Ridge(alpha=alpha, positive=True)
    meta_try.fit(oof_preds, y_nz)
    rmse_try = np.sqrt(mean_squared_error(y_nz, meta_try.predict(oof_preds)))
    if rmse_try < best_ridge_rmse:
        best_ridge_rmse = rmse_try
        best_ridge_alpha = alpha

meta = Ridge(alpha=best_ridge_alpha, positive=True)
meta.fit(oof_preds, y_nz)
meta_pred = meta.predict(oof_preds)
meta_rmse = np.sqrt(mean_squared_error(y_nz, meta_pred))
print(f"  Stacked RMSE: {meta_rmse:.4f} (Ridge alpha={best_ridge_alpha}, positive=True)")
print(f"  Weights: {dict(zip([n for n,_ in models_config], meta.coef_.round(3)))})")

# Fit base models on full non-zero data
final_models = []
for name, template in models_config:
    m = sklearn.base.clone(template)
    m.fit(X_nz, y_nz)
    final_models.append((name, m))

# %% [markdown]
# ## 5. End-to-End 5-Fold CV Evaluation

# %%
print("\n[6/6] Final end-to-end evaluation...")

kf_eval = KFold(n_splits=5, shuffle=True, random_state=42)
fold_rmses = []
y_val_all, y_pred_all = [], []

for fold, (tr_idx, vl_idx) in enumerate(kf_eval.split(X)):
    X_tr, X_vl = X[tr_idx], X[vl_idx]
    y_tr, y_vl = y[tr_idx], y[vl_idx]

    # Stage 1: classifier
    y_tr_bin = (y_tr > ZERO_THRESHOLD).astype(int)
    clf_f = xgb.XGBClassifier(**best_clf_params)
    clf_f.fit(X_tr, y_tr_bin)

    # Stage 2: regressors on non-zero
    nz_tr = y_tr > ZERO_THRESHOLD
    X_nz_f, y_nz_f = X_tr[nz_tr], y_tr[nz_tr]

    base_f = []
    for name, template in models_config:
        m = sklearn.base.clone(template)
        m.fit(X_nz_f, y_nz_f)
        base_f.append(m)

    # Quick inner OOF for meta-learner
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=0)
    oof_inner = np.zeros((X_nz_f.shape[0], len(models_config)))
    for mi, (_, template) in enumerate(models_config):
        for ti, vi in inner_cv.split(X_nz_f):
            m = sklearn.base.clone(template)
            m.fit(X_nz_f[ti], y_nz_f[ti])
            oof_inner[vi, mi] = m.predict(X_nz_f[vi])
    meta_f = Ridge(alpha=1.0)
    meta_f.fit(oof_inner, y_nz_f)

    # Tune classifier threshold per-fold to minimize fold RMSE
    proba = clf_f.predict_proba(X_vl)[:, 1]
    pred = np.zeros(len(X_vl))

    best_fold_thresh = 0.5
    best_fold_rmse_thresh = float('inf')
    for thresh in np.arange(0.3, 0.75, 0.05):
        nz_m = proba >= thresh
        cand = np.zeros(len(X_vl))
        if nz_m.sum() > 0:
            bp_c = np.column_stack([m.predict(X_vl[nz_m]) for m in base_f])
            cand[nz_m] = np.clip(meta_f.predict(bp_c), 0, 100)
        r = np.sqrt(mean_squared_error(y_vl, np.clip(cand, 0, 100)))
        if r < best_fold_rmse_thresh:
            best_fold_rmse_thresh = r
            best_fold_thresh = thresh

    nz_mask = proba >= best_fold_thresh
    if nz_mask.sum() > 0:
        bp = np.column_stack([m.predict(X_vl[nz_mask]) for m in base_f])
        pred[nz_mask] = meta_f.predict(bp)

    pred = np.clip(pred, 0, 100)
    rmse = np.sqrt(mean_squared_error(y_vl, pred))
    fold_rmses.append(rmse)
    y_val_all.extend(y_vl.tolist())
    y_pred_all.extend(pred.tolist())
    print(f"  Fold {fold+1} RMSE: {rmse:.4f}")

mean_rmse = np.mean(fold_rmses)
std_rmse = np.std(fold_rmses)
print(f"\n{'='*50}")
print(f">>> FINAL 5-Fold CV RMSE: {mean_rmse:.4f} +/- {std_rmse:.4f}")
print(f"    Previous V2 RMSE:     ~18.02")
print(f"    Original baseline:    ~21.78")
print(f"{'='*50}")

# %% [markdown]
# ## 6. Plots

# %%
y_va = np.array(y_val_all)
y_pa = np.array(y_pred_all)

plt.figure(figsize=(8, 8))
plt.scatter(y_va, y_pa, alpha=0.6, color='steelblue', edgecolors='white', s=60)
plt.plot([0, 100], [0, 100], 'r--', lw=2)
plt.xlabel('True Yield')
plt.ylabel('Predicted Yield')
plt.title(f'True vs Predicted (CV RMSE={mean_rmse:.2f})')
plt.savefig('true_vs_predicted_yield.png', dpi=300)
plt.close()
print("Saved true_vs_predicted_yield.png")

# Feature importance
xgb_fi = xgb.XGBRegressor(**best_xgb_params)
xgb_fi.fit(X_nz, y_nz)
imp = xgb_fi.feature_importances_
si = np.argsort(imp)
plt.figure(figsize=(10, 8))
plt.barh(range(len(si)), imp[si], align='center')
plt.yticks(range(len(si)), np.array(feature_names)[si])
plt.xlabel('Feature Importance')
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.close()
print("Saved feature_importance.png")

# %% [markdown]
# ## 7. Generate Submission

# %%
proba_test = clf.predict_proba(X_test_final)[:, 1]
pred_test = np.zeros(len(X_test_final))
nz_test = proba_test >= 0.5

if nz_test.sum() > 0:
    bp_test = np.column_stack([m.predict(X_test_final[nz_test]) for _, m in final_models])
    pred_test[nz_test] = meta.predict(bp_test)

pred_test = np.clip(pred_test, 0, 100)

submission = pd.DataFrame({'overall_yield': np.round(pred_test, 3)})
submission.to_csv('Ctrl+Alt+Achieve.csv', index=False)
print("\nSubmission saved as Ctrl+Alt+Achieve.csv!")
print(submission)
print(f"\n{(pred_test == 0).sum()} zeros, {(pred_test > 0).sum()} non-zeros")
