import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, GradientBoostingClassifier, HistGradientBoostingRegressor, ExtraTreesRegressor
import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv("train_dataset.csv")

def feature_engineering(df):
    df_engineered = df.copy()
    df_engineered['residence_time'] = df_engineered['length_m'] / df_engineered['flow_rate_L_min']
    df_engineered['temp_delta'] = df_engineered['jacket_temperature_K'] - df_engineered['inlet_temperature_K']
    df_engineered['arrhenius_inlet'] = np.exp(-5000.0 / df_engineered['inlet_temperature_K'])
    df_engineered['arrhenius_jacket'] = np.exp(-5000.0 / df_engineered['jacket_temperature_K'])
    df_engineered['conc_x_residence'] = df_engineered['concentration_mol_L'] * df_engineered['residence_time']
    df_engineered['volume_throughput'] = df_engineered['length_m'] * df_engineered['flow_rate_L_min']
    df_engineered['residence_time_sq'] = df_engineered['residence_time'] ** 2
    return df_engineered

X = feature_engineering(df_train.drop(columns=['overall_yield']))
y = df_train['overall_yield']

for threshold in [0.0, 1.0, 5.0, 10.0]:
    for rs in [42, 100, 2024]:
        kf = KFold(n_splits=5, shuffle=True, random_state=rs)
        oof_preds = np.zeros(len(X))
        
        for train_idx, val_idx in kf.split(X):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            y_train_class = (y_train > threshold).astype(int)
            clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
            clf.fit(X_train, y_train_class)
            
            mask = y_train > threshold
            X_train_sub, y_train_sub = X_train[mask], y_train[mask]
            
            regressors = [
                GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=42),
                HistGradientBoostingRegressor(max_iter=150, max_depth=4, random_state=42),
                RandomForestRegressor(n_estimators=150, max_depth=5, random_state=42),
                ExtraTreesRegressor(n_estimators=150, max_depth=5, random_state=42)
            ]
            
            fitted_regs = []
            reg_rmses = []
            p_success_val = clf.predict_proba(X_val)[:, 1]
            
            for reg in regressors:
                reg.fit(X_train_sub, y_train_sub)
                fitted_regs.append(reg)
                raw_preds = np.clip(reg.predict(X_val), 0.0, 100.0)
                final_preds = raw_preds * p_success_val
                reg_rmses.append(np.sqrt(mean_squared_error(y_val, final_preds)))
                
            inv_rmses = [1.0 / (rmse + 1e-6) for rmse in reg_rmses]
            weights = [w / sum(inv_rmses) for w in inv_rmses]
            
            val_ens_preds = np.zeros(len(X_val))
            for reg, w in zip(fitted_regs, weights):
                val_ens_preds += w * np.clip(reg.predict(X_val), 0.0, 100.0)
                
            oof_preds[val_idx] = val_ens_preds * p_success_val
            
        cv_rmse = np.sqrt(mean_squared_error(y, oof_preds))
        print(f"Threshold: {threshold} | RS: {rs} | RMSE: {cv_rmse:.4f}")
