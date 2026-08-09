# %% [markdown]
# # Predictive Modeling Optimization Challenge - Model Training
# 
# ## 1. Introduction
# In this notebook, we build a predictive model to predict the `overall_yield` of Product B based on the operating conditions. We will use a Gradient Boosting model (XGBoost/HistGradientBoosting) which is well-suited for capturing complex non-linear relationships in tabular data.

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
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
# ## 3. Data Preparation
# Split the training data into features (X) and target (y).

# %%
X = df_train.drop(columns=['overall_yield'])
y = df_train['overall_yield']
X_test_final = df_test.copy()

# Split train data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
print(f"Val shapes: X={X_val.shape}, y={y_val.shape}")

# %% [markdown]
# ## 4. Model Selection and Training
# Given the small dataset (150 rows), a Random Forest and Gradient Boosting model are strong candidates. We'll train a GradientBoostingRegressor.

# %%
# Initialize model
gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
    subsample=0.8
)

# Train the model
gb_model.fit(X_train, y_train)

# Predict on validation set
y_val_pred = gb_model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"Validation RMSE: {val_rmse:.4f}")

# %% [markdown]
# ## 5. Model Evaluation & Visualization
# Let's visualize the True vs Predicted yields on the validation set.

# %%
plt.figure(figsize=(8, 8))
plt.scatter(y_val, y_val_pred, alpha=0.7, color='b')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('True Yield')
plt.ylabel('Predicted Yield')
plt.title('True vs Predicted Yield (Validation Set)')
plt.savefig('true_vs_predicted_yield.png', dpi=300)
plt.close()

# %% [markdown]
# ### Feature Importance
# Let's see which features the model relies on the most. This provides Process Insight.

# %%
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
# To maximize performance, we retrain the model on the entire training set before predicting on the blind test set.

# %%
final_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
    subsample=0.8
)
final_model.fit(X, y)

# Predict on test set
final_predictions = final_model.predict(X_test_final)

# %% [markdown]
# ## 7. Prepare Submission File
# Generate the submission CSV containing exactly 50 rows and one column `overall_yield`, rounded to 3 decimal places.

# %%
submission = pd.DataFrame({'overall_yield': np.round(final_predictions, 3)})
submission.to_csv('Ctrl+Alt+Achieve.csv', index=False)

print("Submission saved successfully as Ctrl+Alt+Achieve.csv!")
print(submission.head())
