# problem 2 (a)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
iris_df = pd.read_csv("iris_data.csv")  # Update the path if needed

# Create scatterplot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=iris_df, x='petal_length', y='petal_width', hue='type')
plt.title('Petal Width vs. Petal Length by Subspecies')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.grid(True)
plt.show()

#problme 2 (b)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Select the numerical features
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = iris_df[features]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

# Add PCA results to DataFrame
iris_df['PC1'] = principal_components[:, 0]
iris_df['PC2'] = principal_components[:, 1]

# Create PCA scatterplot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=iris_df, x='PC1', y='PC2', hue='type')
plt.title('PCA: First Two Principal Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()

# problem 2 (c)
# Display explained variance
explained_variance = pca.explained_variance_ratio_
print("Proportion of Variance Explained:")
print(f"PC1: {explained_variance[0]:.4f}")
print(f"PC2: {explained_variance[1]:.4f}")
print(f"Total: {explained_variance[0] + explained_variance[1]:.4f}")

# problem 2 (d)
# Loadings matrix
loadings = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2'],
    index=features
)

print("Loadings for original features:")
print(loadings)
