# %% [markdown]
# # Predictive Modeling Optimization Challenge - Model Training
# 
# ## 1. Introduction
# In this notebook, we build a robust predictive model to predict the `overall_yield` of Product B. To improve the RMSE on a small dataset, we will implement domain-specific feature engineering, hyperparameter tuning, and ensemble modeling.

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
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
# ## 3. Feature Engineering
# We will define a function to create chemistry-based features to help the models mimic physics laws.

# %%
def feature_engineering(df):
    df_engineered = df.copy()
    # 1. Temperature Delta (Driving force for heat transfer)
    df_engineered['delta_temp'] = df_engineered['jacket_temperature_K'] - df_engineered['inlet_temperature_K']
    
    # 2. Inverse Temperature (Arrhenius relationship 1/T)
    df_engineered['inv_jacket_temp'] = 1.0 / df_engineered['jacket_temperature_K']
    df_engineered['inv_inlet_temp'] = 1.0 / df_engineered['inlet_temperature_K']
    
    # 3. Residence Time Proxy (length / flow_rate)
    df_engineered['residence_time_proxy'] = df_engineered['length_m'] / df_engineered['flow_rate_L_min']
    
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
# ## 4. Ensemble Modeling
# We will combine Gradient Boosting, Random Forest, and SVR to create a robust Voting Regressor. SVR benefits from scaling, so we put it in a Pipeline.

# %%
# Define individual models
gb_model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    random_state=42
)

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=5,
    random_state=42
)

svr_model = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(kernel='rbf', C=100, epsilon=0.1))
])

# Create an Ensemble Model
ensemble_model = VotingRegressor([
    ('gb', gb_model),
    ('rf', rf_model),
    ('svr', svr_model)
])

# Cross-validation score
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(ensemble_model, X, y, cv=kf, scoring='neg_root_mean_squared_error')
cv_rmse = -cv_scores.mean()

print(f"Cross-Validation RMSE (Ensemble): {cv_rmse:.4f}")

# Fit on training fold and validate
ensemble_model.fit(X_train, y_train)
y_val_pred = ensemble_model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"Hold-out Validation RMSE: {val_rmse:.4f}")

# %% [markdown]
# ## 5. Model Evaluation & Visualization

# %%
plt.figure(figsize=(8, 8))
plt.scatter(y_val, y_val_pred, alpha=0.7, color='b')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('True Yield')
plt.ylabel('Predicted Yield')
plt.title('True vs Predicted Yield (Ensemble Model)')
plt.savefig('true_vs_predicted_yield.png', dpi=300)
plt.close()

# %% [markdown]
# ### Feature Importance (From Gradient Boosting)

# %%
gb_model.fit(X, y)
feature_importance = gb_model.feature_importances_
sorted_idx = np.argsort(feature_importance)

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(X.columns)[sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Gradient Boosting - Feature Importance')
plt.savefig('feature_importance.png', dpi=300)
plt.close()

# %% [markdown]
# ## 6. Retrain on Full Data and Predict for Test Set

# %%
# Fit the ensemble model on ALL available training data
ensemble_model.fit(X, y)

# Predict on test set
final_predictions = ensemble_model.predict(X_test_final)

# %% [markdown]
# ## 7. Prepare Submission File
# Generate the submission CSV containing exactly 50 rows and one column `overall_yield`, rounded to 3 decimal places.

# %%
submission = pd.DataFrame({'overall_yield': np.round(final_predictions, 3)})
submission.to_csv('Ctrl+Alt+Achieve.csv', index=False)

print("Submission saved successfully as Ctrl+Alt+Achieve.csv!")
print(submission.head())
