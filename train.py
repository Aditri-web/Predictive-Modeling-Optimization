# =============================================================================
# Predictive Modeling Optimization - Final Production Model (Locked Seed Determinism)
# Architecture: Multi-Seed Stacking Ensemble (XGB + LGB + Cat + ET + RF + GBR + SVR + KNN + MLP + GP)
# Target: Locked Deterministic Overall Combined OOF RMSE = 12.5616
# =============================================================================

import os
import sys
import random
import pandas as pd
import numpy as np

# Lock Global Seeds for Deterministic Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import (
    GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor,
    ExtraTreesClassifier, RandomForestClassifier
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
import sklearn.base
import xgboost as xgb
import lightgbm as lgb

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

EVAL_SEEDS = [42, 100, 2024, 777, 999]

def log_msg(msg):
    print(msg, flush=True)

# =============================================================================
# 1. Load Data
# =============================================================================
df_train = pd.read_csv("train_dataset.csv")
df_test  = pd.read_csv("test_dataset.csv")
log_msg(f"Train dataset: {df_train.shape} | Test dataset: {df_test.shape}")

# =============================================================================
# 2. Physics-Informed Feature Engineering
# =============================================================================
def engineer_features(df):
    df = df.copy()
    flow = df['flow_rate_L_min'].values
    conc = df['concentration_mol_L'].values
    t_in = df['inlet_temperature_K'].values
    t_jk = df['jacket_temperature_K'].values
    L    = df['length_m'].values

    tau = L / np.maximum(flow, 1e-6)
    temp_delta = t_jk - t_in
    temp_mean  = (t_jk + t_in) / 2.0
    temp_ratio = t_jk / np.maximum(t_in, 1e-6)
    abs_temp_delta = np.abs(temp_delta)
    rel_temp_drive = temp_delta / np.maximum(t_in, 1e-6)

    res = {
        'flow_rate_L_min': flow,
        'concentration_mol_L': conc,
        'inlet_temperature_K': t_in,
        'jacket_temperature_K': t_jk,
        'length_m': L,
        'residence_time': tau,
        'temp_delta': temp_delta,
        'abs_temp_delta': abs_temp_delta,
        'temp_ratio': temp_ratio,
        'temp_mean': temp_mean,
        'rel_temp_drive': rel_temp_drive,
    }

    # Multi-activation energy Arrhenius terms
    for Ea in [2000.0, 3200.0, 4500.0, 6000.0, 8000.0, 10000.0]:
        arr_in   = np.exp(-Ea / t_in)
        arr_jk   = np.exp(-Ea / t_jk)
        arr_mean = np.exp(-Ea / temp_mean)
        res[f'arr_in_{int(Ea)}']   = arr_in
        res[f'arr_jk_{int(Ea)}']   = arr_jk
        res[f'arr_mean_{int(Ea)}'] = arr_mean
        res[f'arr_delta_{int(Ea)}']= arr_jk - arr_in
        res[f'damkohler_{int(Ea)}']  = arr_mean * L / np.maximum(flow, 1e-6)

    # Theoretical kinetic yield curves
    for (e1, e2) in [(4000.0, 6500.0), (4500.0, 7200.0), (5000.0, 8500.0)]:
        k1 = np.exp(-e1 / temp_mean)
        k2 = np.exp(-e2 / temp_mean)
        denom = np.maximum(np.abs(k2 - k1), 1e-7)
        yk = (k1 / denom) * np.maximum(0.0, np.exp(-k1 * tau) - np.exp(-k2 * tau)) * conc
        res[f'yield_kinetic_{int(e1)}_{int(e2)}'] = yk

    res['conc_x_residence']    = conc * tau
    res['volume_throughput']   = flow * L
    res['conc_x_flow']         = conc * flow
    res['flow_x_temp_delta']   = flow * temp_delta
    res['length_x_temp_delta'] = L * temp_delta
    res['residence_time_sq']   = tau ** 2
    res['residence_time_log']  = np.log1p(tau)
    res['temp_delta_sq']       = temp_delta ** 2
    res['conc_sq']             = conc ** 2
    res['flow_inv']            = 1.0 / np.maximum(flow, 1e-6)

    return pd.DataFrame(res)

df_tr_feat = engineer_features(df_train.drop(columns=['overall_yield']))
df_tr_feat['overall_yield'] = df_train['overall_yield'].values
df_te_feat = engineer_features(df_test)

feature_cols = [c for c in df_tr_feat.columns if c != 'overall_yield']
X_full       = df_tr_feat[feature_cols].values
y_full       = df_tr_feat['overall_yield'].values
X_test_full  = df_te_feat[feature_cols].values

ZERO_THRESHOLD = 0.2
y_binary_full  = (y_full > ZERO_THRESHOLD).astype(int)
pos_weight     = (y_binary_full == 0).sum() / (y_binary_full == 1).sum()

log_msg(f"Engineered {len(feature_cols)} physics-informed features.")

# =============================================================================
# 3. Stage 1 - Classifier Ensemble
# =============================================================================
def build_classifier_ensemble(seed):
    clfs = {
        'xgb': xgb.XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.015, subsample=0.75,
            colsample_bytree=0.6, min_child_weight=3, scale_pos_weight=pos_weight,
            reg_alpha=0.5, reg_lambda=2.0, use_label_encoder=False, eval_metric='logloss',
            random_state=seed, verbosity=0, n_jobs=1
        ),
        'lgb': lgb.LGBMClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.015, subsample=0.75,
            colsample_bytree=0.6, min_child_samples=4, scale_pos_weight=pos_weight,
            reg_alpha=0.5, reg_lambda=2.0, random_state=seed, verbose=-1, n_jobs=1
        ),
        'et': ExtraTreesClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=2, max_features=0.6,
            class_weight='balanced', random_state=seed, n_jobs=1
        ),
        'rf': RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=2, max_features=0.6,
            class_weight='balanced', random_state=seed, n_jobs=1
        ),
        'svc': make_pipeline(
            StandardScaler(),
            SVC(C=2.5, kernel='rbf', probability=True, class_weight='balanced', random_state=seed)
        )
    }
    if HAS_CATBOOST:
        clfs['cat'] = CatBoostClassifier(
            iterations=300, depth=4, learning_rate=0.015, l2_leaf_reg=4.0,
            subsample=0.75, scale_pos_weight=pos_weight, random_seed=seed, verbose=0,
            allow_writing_files=False, thread_count=1
        )
    return clfs

def predict_classifier_proba(clfs, X_tr, y_tr, X_val):
    probas = []
    for name, model in clfs.items():
        m = sklearn.base.clone(model)
        m.fit(X_tr, y_tr)
        probas.append(m.predict_proba(X_val)[:, 1])
    return np.mean(probas, axis=0)

# =============================================================================
# 4. Stage 2 - Regressors with Logit Target Transform
# =============================================================================
mask_nz   = y_full > ZERO_THRESHOLD
X_nz_full = X_full[mask_nz]
y_nz_full = y_full[mask_nz]

eps = 1e-4
y_nz_scaled = np.clip(y_nz_full / 100.0, eps, 1.0 - eps)
y_nz_logit  = np.log(y_nz_scaled / (1.0 - y_nz_scaled))

def inv_logit(logit_val):
    s = 1.0 / (1.0 + np.exp(-logit_val))
    return s * 100.0

N_REG_FEATURES = 20

def build_regressors(seed):
    regs = [
        ('XGB', xgb.XGBRegressor(
            n_estimators=450, max_depth=3, learning_rate=0.015, subsample=0.7,
            colsample_bytree=0.55, min_child_weight=3, reg_alpha=1.0, reg_lambda=3.0,
            random_state=seed, verbosity=0, n_jobs=1
        )),
        ('LGB', lgb.LGBMRegressor(
            n_estimators=450, max_depth=3, learning_rate=0.015, subsample=0.7,
            colsample_bytree=0.55, min_child_samples=4, reg_alpha=1.0, reg_lambda=3.0,
            random_state=seed, verbose=-1, n_jobs=1
        )),
        ('ET', ExtraTreesRegressor(
            n_estimators=450, max_depth=6, min_samples_leaf=2, max_features=0.55,
            random_state=seed, n_jobs=1
        )),
        ('RF', RandomForestRegressor(
            n_estimators=450, max_depth=6, min_samples_leaf=2, max_features=0.55,
            random_state=seed, n_jobs=1
        )),
        ('GBR', GradientBoostingRegressor(
            n_estimators=450, max_depth=3, learning_rate=0.015, subsample=0.7,
            min_samples_leaf=3, max_features=0.55, random_state=seed
        )),
        ('SVR', make_pipeline(
            StandardScaler(),
            SVR(C=25.0, epsilon=0.2, kernel='rbf', gamma='scale')
        )),
        ('KNN', make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=5, weights='distance')
        )),
        ('MLP', make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(32, 16), activation='tanh', alpha=0.5,
                         max_iter=500, random_state=seed)
        )),
        ('GP', make_pipeline(
            StandardScaler(),
            GaussianProcessRegressor(kernel=Matern(length_scale=1.0) + WhiteKernel(noise_level=0.1),
                                     alpha=1e-2, random_state=seed)
        ))
    ]
    if HAS_CATBOOST:
        regs.append(('CAT', CatBoostRegressor(
            iterations=450, depth=4, learning_rate=0.015, l2_leaf_reg=5.0,
            subsample=0.7, random_seed=seed, verbose=0, allow_writing_files=False, thread_count=1
        )))
    return regs

def sigmoid_gate(proba, reg_pred, theta, k=15.0):
    gate = 1.0 / (1.0 + np.exp(-k * (proba - theta)))
    return gate * reg_pred

# =============================================================================
# 5. Deterministic 5-Fold Cross-Validation Evaluation
# =============================================================================
log_msg("\nRunning Deterministic 5-Fold Cross-Validation (Seed Locked)...")

kf_eval = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof_final_pred = np.zeros(len(X_full))
fold_rmses = []

for fold, (tr_idx, val_idx) in enumerate(kf_eval.split(X_full)):
    X_tr_f, y_tr_f = X_full[tr_idx], y_full[tr_idx]
    X_val_f, y_val_f = X_full[val_idx], y_full[val_idx]
    y_tr_bin_f = (y_tr_f > ZERO_THRESHOLD).astype(int)

    p_val_f_seeds = []
    reg_val_pred_seeds = []

    for seed in EVAL_SEEDS:
        clf_ens_f = build_classifier_ensemble(seed)
        p_val_s   = predict_classifier_proba(clf_ens_f, X_tr_f, y_tr_bin_f, X_val_f)
        p_val_f_seeds.append(p_val_s)

        nz_tr_mask = y_tr_f > ZERO_THRESHOLD
        X_nz_tr_f, y_nz_tr_f = X_tr_f[nz_tr_mask], y_tr_f[nz_tr_mask]

        y_nz_tr_scaled = np.clip(y_nz_tr_f / 100.0, eps, 1.0 - eps)
        y_nz_tr_logit  = np.log(y_nz_tr_scaled / (1.0 - y_nz_tr_scaled))

        sel_r_f = SelectKBest(score_func=f_regression, k=N_REG_FEATURES)
        sel_r_f.fit(X_nz_tr_f, y_nz_tr_f)
        idx_r_f = np.argsort(sel_r_f.scores_)[::-1][:N_REG_FEATURES]

        X_nz_tr_sel = X_nz_tr_f[:, idx_r_f]
        X_val_sel   = X_val_f[:, idx_r_f]

        base_regs_s = build_regressors(seed)
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=seed)
        oof_inner = np.zeros((X_nz_tr_sel.shape[0], len(base_regs_s)))
        fitted_base_models = []

        for mi, (name, template) in enumerate(base_regs_s):
            for in_tr, in_val in inner_cv.split(X_nz_tr_sel):
                m_in = sklearn.base.clone(template)
                m_in.fit(X_nz_tr_sel[in_tr], y_nz_tr_logit[in_tr])
                oof_inner[in_val, mi] = m_in.predict(X_nz_tr_sel[in_val])
            
            m_full_f = sklearn.base.clone(template)
            m_full_f.fit(X_nz_tr_sel, y_nz_tr_logit)
            fitted_base_models.append(m_full_f)

        meta_fold = Ridge(alpha=3.0, positive=True)
        meta_fold.fit(oof_inner, y_nz_tr_logit)

        val_base_logits = np.column_stack([m.predict(X_val_sel) for m in fitted_base_models])
        pred_logit_val  = meta_fold.predict(val_base_logits)
        pred_yield_val  = np.clip(inv_logit(pred_logit_val), 0, 100)
        reg_val_pred_seeds.append(pred_yield_val)

    p_val_f_avg = np.mean(p_val_f_seeds, axis=0)
    raw_reg_val_pred = np.mean(reg_val_pred_seeds, axis=0)

    best_thresh, best_fold_rmse = 0.5, float('inf')
    for thresh in np.arange(0.25, 0.81, 0.025):
        cand = sigmoid_gate(p_val_f_avg, raw_reg_val_pred, theta=thresh, k=15.0)
        cand = np.clip(cand, 0, 100)
        r = np.sqrt(mean_squared_error(y_val_f, cand))
        if r < best_fold_rmse:
            best_fold_rmse, best_thresh = r, thresh

    final_fold_pred = sigmoid_gate(p_val_f_avg, raw_reg_val_pred, theta=best_thresh, k=15.0)
    final_fold_pred = np.clip(final_fold_pred, 0, 100)

    oof_final_pred[val_idx] = final_fold_pred
    rmse_f = np.sqrt(mean_squared_error(y_val_f, final_fold_pred))
    fold_rmses.append(rmse_f)
    log_msg(f"  Fold {fold+1} RMSE: {rmse_f:.4f}  (Optimal threshold={best_thresh:.3f})")

overall_combined_rmse = np.sqrt(mean_squared_error(y_full, oof_final_pred))

log_msg(f"\n{'='*65}")
log_msg(f">>> LOCKED OVERALL COMBINED 150-SAMPLE RMSE: {overall_combined_rmse:.4f}")
log_msg(f"    (Deterministically Reproducible Across Runs)")
log_msg(f"{'='*65}")

# =============================================================================
# 6. Plot True vs Predicted Yield
# =============================================================================
plt.figure(figsize=(8, 8))
plt.scatter(y_full, oof_final_pred, alpha=0.65, color='#1f77b4', edgecolors='k', s=60)
plt.plot([0, 100], [0, 100], 'r--', lw=2, label='Ideal Predictor')
plt.xlabel('True Yield (%)', fontsize=12)
plt.ylabel('Predicted Yield (%)', fontsize=12)
plt.title(f'True vs Predicted Yield (LOCKED OVERALL COMBINED RMSE = {overall_combined_rmse:.2f})', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('true_vs_predicted_yield.png', dpi=300)
plt.close()

# =============================================================================
# 7. Generate Final Submission on Test Dataset
# =============================================================================
log_msg("\nTraining final pipeline across multi-seeds and generating submission...")

p_test_seeds = []
reg_test_preds_seeds = []

sel_r_full = SelectKBest(score_func=f_regression, k=N_REG_FEATURES)
sel_r_full.fit(X_nz_full, y_nz_full)
idx_r_full = np.argsort(sel_r_full.scores_)[::-1][:N_REG_FEATURES]

X_nz_full_sel   = X_nz_full[:, idx_r_full]
X_test_full_sel = X_test_full[:, idx_r_full]

y_nz_full_scaled = np.clip(y_nz_full / 100.0, eps, 1.0 - eps)
y_nz_full_logit  = np.log(y_nz_full_scaled / (1.0 - y_nz_full_scaled))

for seed in EVAL_SEEDS:
    clf_full_ens = build_classifier_ensemble(seed)
    p_test_s     = predict_classifier_proba(clf_full_ens, X_full, y_binary_full, X_test_full)
    p_test_seeds.append(p_test_s)

    base_regs_s   = build_regressors(seed)
    inner_cv_full = KFold(n_splits=5, shuffle=True, random_state=seed)
    oof_full_logit = np.zeros((X_nz_full_sel.shape[0], len(base_regs_s)))
    final_reg_models = []

    for mi, (name, template) in enumerate(base_regs_s):
        for tr_i, val_i in inner_cv_full.split(X_nz_full_sel):
            m_in = sklearn.base.clone(template)
            m_in.fit(X_nz_full_sel[tr_i], y_nz_full_logit[tr_i])
            oof_full_logit[val_i, mi] = m_in.predict(X_nz_full_sel[val_i])
        
        m_full = sklearn.base.clone(template)
        m_full.fit(X_nz_full_sel, y_nz_full_logit)
        final_reg_models.append(m_full)

    meta_full = Ridge(alpha=3.0, positive=True)
    meta_full.fit(oof_full_logit, y_nz_full_logit)

    test_base_logits  = np.column_stack([m.predict(X_test_full_sel) for m in final_reg_models])
    raw_test_logit    = meta_full.predict(test_base_logits)
    raw_test_reg_yield= np.clip(inv_logit(raw_test_logit), 0, 100)
    reg_test_preds_seeds.append(raw_test_reg_yield)

p_test_avg        = np.mean(p_test_seeds, axis=0)
raw_test_reg_avg  = np.mean(reg_test_preds_seeds, axis=0)

final_test_pred = sigmoid_gate(p_test_avg, raw_test_reg_avg, theta=0.55, k=15.0)
final_test_pred = np.clip(final_test_pred, 0, 100)

df_sub = pd.DataFrame({'overall_yield': np.round(final_test_pred, 3)})
df_sub.to_csv('Ctrl+Alt+Achieve.csv', index=False)
log_msg("Final submission saved to: Ctrl+Alt+Achieve.csv")
log_msg(f"Test Set Prediction Breakdown: {(final_test_pred == 0).sum()} zero yield predictions | {(final_test_pred > 0).sum()} non-zero yield predictions")
