# K-Nearest Neighbors (KNN) Classification on Iris Dataset

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from matplotlib.colors import ListedColormap

# Step 2: Load the dataset
df = pd.read_csv("Iris.csv")  # Make sure file is in same folder
print(df.head())

# Step 3: Features & Target
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

# Step 4: Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Step 6: Try different K values
print("\nTesting different K values:")
for k in [1, 3, 5, 7, 9]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(f"K={k} -> Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Step 7: Final model with best K
best_k = 5
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Step 8: Evaluation
print("\nFinal Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 9: Plot confusion matrix heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=knn.classes_, yticklabels=knn.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix (K={best_k})")
plt.show()

# Step 10: Decision boundary plot (using first 2 features)
# Encode labels as numbers for contour plot
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Only take first 2 features for visualization
X_plot = X_scaled[:, :2]
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_plot, y_encoded, test_size=0.2, random_state=42
)

knn_plot = KNeighborsClassifier(n_neighbors=best_k)
knn_plot.fit(X_train_p, y_train_p)

# Color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ['red', 'green', 'blue']

# Create meshgrid
x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

# Predictions over grid
Z = knn_plot.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundaries
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)

# Scatter plot of data points
sns.scatterplot(
    x=X_plot[:, 0], y=X_plot[:, 1],
    hue=le.inverse_transform(y_encoded),
    palette=cmap_bold, s=50, edgecolor='k'
)

plt.title(f"KNN Decision Boundaries (K={best_k}) - First 2 Features")
plt.xlabel("Sepal Length (Standardized)")
plt.ylabel("Sepal Width (Standardized)")
plt.show()
