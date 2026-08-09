import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, LogisticRegression
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier

# Load data
df_train = pd.read_csv("train_dataset.csv")

def engineer_features(df):
    df = df.copy()
    flow   = df['flow_rate_L_min']
    conc   = df['concentration_mol_L']
    t_in   = df['inlet_temperature_K']
    t_jk   = df['jacket_temperature_K']
    length = df['length_m']
    df['residence_time']       = length / flow
    df['temp_delta']           = t_jk - t_in
    df['temp_mean']            = (t_jk + t_in) / 2
    for Ea in [3000, 5000, 8000, 12000]:
        df[f'arrh_mean_{Ea}']  = np.exp(-Ea / df['temp_mean'])
    df['residence_time_log']   = np.log1p(df['residence_time'])
    df['conc_x_residence']     = conc * df['residence_time']
    return df

df_train_feat = engineer_features(df_train.drop(columns=['overall_yield']))
X = df_train_feat.values
y = df_train['overall_yield'].values
y_class = (y > 0).astype(int) # 1 if non-zero, 0 if zero

kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_rmses = []
for fold, (tr_idx, vl_idx) in enumerate(kf.split(X)):
    X_tr, X_vl = X[tr_idx], X[vl_idx]
    y_tr, y_vl = y[tr_idx], y[vl_idx]
    yc_tr, yc_vl = y_class[tr_idx], y_class[vl_idx]

    # Train Classifier (Soft Probability)
    clf = CatBoostClassifier(random_state=42, iterations=300, depth=4, verbose=0, auto_class_weights='Balanced')
    clf.fit(X_tr, yc_tr)
    prob_vl = clf.predict_proba(X_vl)[:, 1] # Probability of being non-zero

    # Train Regressor (on ALL data or just positive data? Let's do ALL data to avoid bias)
    reg = CatBoostRegressor(random_state=42, iterations=400, depth=4, verbose=0)
    reg.fit(X_tr, y_tr)
    reg_pred_vl = reg.predict(X_vl)

    # Soft Cascade
    final_pred = np.clip(prob_vl * reg_pred_vl, 0, 100)

    # Hard Cascade for comparison
    hard_pred = np.clip((prob_vl > 0.5) * reg_pred_vl, 0, 100)

    # Standard Regressor for comparison
    std_pred = np.clip(reg_pred_vl, 0, 100)

    print(f"Fold {fold}: Soft RMSE: {np.sqrt(mean_squared_error(y_vl, final_pred)):.4f} | Hard: {np.sqrt(mean_squared_error(y_vl, hard_pred)):.4f} | Std: {np.sqrt(mean_squared_error(y_vl, std_pred)):.4f}")
    fold_rmses.append(np.sqrt(mean_squared_error(y_vl, final_pred)))

print(f"Soft Cascade CV RMSE: {np.mean(fold_rmses):.4f}")
