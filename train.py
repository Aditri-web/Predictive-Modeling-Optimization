# %% [markdown]
# # Predictive Modeling Optimization Challenge - Model Training
# 
# ## 1. Introduction
# In this notebook, we build a robust predictive model to predict the `overall_yield` of Product B. Following physical constraints and the bimodal nature of the yield (success vs failure cliffs), we utilize a Two-Stage Pipeline (Classifier + Regressor), 7 physics-informed features, and a dynamically weighted ensemble.

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, GradientBoostingClassifier, HistGradientBoostingRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

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
    
    # 1. Residence Time: Core reactor variable
    df_engineered['residence_time'] = df_engineered['length_m'] / df_engineered['flow_rate_L_min']
    
    # 2. Temperature Delta: Heat exchange driving force
    df_engineered['temp_delta'] = df_engineered['jacket_temperature_K'] - df_engineered['inlet_temperature_K']
    
    # 3. Arrhenius Terms: exp(-5000 / T)
    df_engineered['arrhenius_inlet'] = np.exp(-5000.0 / df_engineered['inlet_temperature_K'])
    df_engineered['arrhenius_jacket'] = np.exp(-5000.0 / df_engineered['jacket_temperature_K'])
    
    # 4. Conversion Capacity
    df_engineered['conc_x_residence'] = df_engineered['concentration_mol_L'] * df_engineered['residence_time']
    
    # 5. Volume Throughput
    df_engineered['volume_throughput'] = df_engineered['length_m'] * df_engineered['flow_rate_L_min']
    
    # 6. Residence Time Squared (over-cooking effect)
    df_engineered['residence_time_sq'] = df_engineered['residence_time'] ** 2
    
    return df_engineered

X_raw = df_train.drop(columns=['overall_yield'])
y = df_train['overall_yield']
X_test_raw = df_test.copy()

X = feature_engineering(X_raw)
X_test_final = feature_engineering(X_test_raw)

# %% [markdown]
# ## 4. Two-Stage Bimodal Pipeline & Weighted Ensemble
# We implement a custom 5-Fold Cross-Validation pipeline. For each fold, we:
# 1. Train a Classifier on (yield > 1.0)
# 2. Train regressors only on successful reactions (yield > 1.0)
# 3. Weight the regressors by inverse RMSE on the holdout fold
# 4. Soft blend: Final Pred = Ensemble_Pred * Classifier_Prob

# %%
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
models_per_fold = []
weights_per_fold = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # Stage 1: Classifier targets
    y_train_class = (y_train > 1.0).astype(int)
    
    clf = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42)
    clf.fit(X_train, y_train_class)
    
    # Stage 2: Regressors on non-zero subset
    mask = y_train > 1.0
    X_train_sub = X_train[mask]
    y_train_sub = y_train[mask]
    
    regressors = [
        GradientBoostingRegressor(n_estimators=150, max_depth=4, random_state=42, subsample=0.8),
        HistGradientBoostingRegressor(max_iter=150, max_depth=4, random_state=42),
        RandomForestRegressor(n_estimators=150, max_depth=5, random_state=42)
    ]
    
    fitted_regs = []
    reg_rmses = []
    
    for reg in regressors:
        reg.fit(X_train_sub, y_train_sub)
        fitted_regs.append(reg)
        
        # Evaluate to find weights using validation set
        # Note: We evaluate RMSE on the full val set using the two-stage logic for accurate weights
        p_success = clf.predict_proba(X_val)[:, 1]
        raw_preds = np.clip(reg.predict(X_val), 0.0, 100.0)
        final_preds = raw_preds * p_success
        fold_rmse = np.sqrt(mean_squared_error(y_val, final_preds))
        reg_rmses.append(fold_rmse)
        
    # Inverse RMSE weighting
    inv_rmses = [1.0 / (rmse + 1e-6) for rmse in reg_rmses]
    weights = [w / sum(inv_rmses) for w in inv_rmses]
    
    # Generate OOF predictions for this fold
    p_success_val = clf.predict_proba(X_val)[:, 1]
    val_ens_preds = np.zeros(len(X_val))
    for reg, w in zip(fitted_regs, weights):
        val_ens_preds += w * np.clip(reg.predict(X_val), 0.0, 100.0)
        
    oof_preds[val_idx] = val_ens_preds * p_success_val
    
    models_per_fold.append((clf, fitted_regs))
    weights_per_fold.append(weights)

cv_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"5-Fold CV RMSE (Two-Stage Pipeline): {cv_rmse:.4f}")

# %% [markdown]
# ## 5. Model Evaluation & Visualization

# %%
plt.figure(figsize=(8, 8))
plt.scatter(y, oof_preds, alpha=0.7, color='b')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('True Yield')
plt.ylabel('Predicted Yield (Two-Stage CV)')
plt.title('True vs Predicted Yield (Out-of-Fold)')
plt.savefig('true_vs_predicted_yield.png', dpi=300)
plt.close()

# %% [markdown]
# ### Feature Importance (From average GBR models)

# %%
# Average feature importances across folds
avg_importance = np.zeros(X.shape[1])
for _, fitted_regs in models_per_fold:
    avg_importance += fitted_regs[0].feature_importances_ / 5.0

sorted_idx = np.argsort(avg_importance)

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), avg_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(X.columns)[sorted_idx])
plt.xlabel('Average Feature Importance')
plt.title('GBR Regressor - Feature Importance')
plt.savefig('feature_importance.png', dpi=300)
plt.close()

# %% [markdown]
# ## 6. Predict for Test Set
# We average the predictions across all 5 folds to create a very robust final test prediction.

# %%
final_test_preds = np.zeros(len(X_test_final))

for fold, (clf, fitted_regs) in enumerate(models_per_fold):
    weights = weights_per_fold[fold]
    p_success = clf.predict_proba(X_test_final)[:, 1]
    
    fold_ens_preds = np.zeros(len(X_test_final))
    for reg, w in zip(fitted_regs, weights):
        fold_ens_preds += w * np.clip(reg.predict(X_test_final), 0.0, 100.0)
        
    final_test_preds += (fold_ens_preds * p_success) / 5.0

# Clip bounds just to be safe
final_test_preds = np.clip(final_test_preds, 0.0, 100.0)

# %% [markdown]
# ## 7. Prepare Submission File
# Generate the submission CSV containing exactly 50 rows and one column `overall_yield`, rounded to 3 decimal places.

# %%
submission = pd.DataFrame({'overall_yield': np.round(final_test_preds, 3)})
submission.to_csv('Ctrl+Alt+Achieve.csv', index=False)

print("Submission saved successfully as Ctrl+Alt+Achieve.csv!")
print(submission.head())
