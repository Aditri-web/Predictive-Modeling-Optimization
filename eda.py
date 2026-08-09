# %% [markdown]
# # Predictive Modeling Optimization Challenge - Exploratory Data Analysis (EDA)
# 
# ## 1. Introduction
# The dataset is derived from a non-isothermal, continuous flow reactor. The reactor involves competing reactions sensitive to operating conditions. 
# We are trying to predict the `overall_yield` of Product B.
# 
# Let's begin by importing necessary libraries and loading the dataset.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# %% [markdown]
# ## 2. Load the Dataset
# We load the training data which contains 150 rows.

# %%
df_train = pd.read_csv("train_dataset.csv")
print(df_train.head())

# %% [markdown]
# ## 3. Data Overview
# Let's check the basic info, missing values, and descriptive statistics.

# %%
df_train.info()

# %%
df_train.describe()

# %% [markdown]
# There are no missing values. The features are numerical, and the ranges vary significantly (e.g., temperatures are around 300-400K, while concentrations are in single digits). We might need feature scaling for some machine learning models.

# %% [markdown]
# ## 4. Univariate Analysis
# Let's look at the distributions of each feature and the target variable.

# %%
features = [col for col in df_train.columns if col != 'overall_yield']
target = 'overall_yield'

fig, axes = plt.subplots(3, 2, figsize=(15, 15))
axes = axes.flatten()

for i, col in enumerate(df_train.columns):
    sns.histplot(df_train[col], kde=True, ax=axes[i], color='blue' if col != target else 'green')
    axes[i].set_title(f'Distribution of {col}')
    
plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=300)
plt.close()

# %% [markdown]
# ## 5. Bivariate Analysis
# Let's explore the relationship between the features and the target variable (`overall_yield`).

# %%
fig, axes = plt.subplots(3, 2, figsize=(15, 15))
axes = axes.flatten()

for i, feature in enumerate(features):
    sns.scatterplot(data=df_train, x=feature, y=target, ax=axes[i], alpha=0.7)
    axes[i].set_title(f'{feature} vs {target}')
    
# Remove empty subplot if features count is odd
if len(features) < len(axes):
    fig.delaxes(axes[-1])

plt.tight_layout()
plt.savefig('features_vs_target.png', dpi=300)
plt.close()

# %% [markdown]
# ## 6. Correlation Analysis
# We analyze the Pearson correlation between variables to see if there are strong linear relationships.

# %%
corr_matrix = df_train.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png', dpi=300)
plt.close()

# %% [markdown]
# ### Observations from EDA:
# 1. The target `overall_yield` has a varied distribution.
# 2. Some features show complex non-linear relationships with the target, as expected from chemical kinetics.
# 3. There is no extreme multicollinearity among input features.
# 
# In the next notebook, we will train a predictive model using this data.
