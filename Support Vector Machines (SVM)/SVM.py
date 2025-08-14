import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# 1. Load dataset
df = pd.read_csv("breast-cancer.csv")

# Inspect
print("Dataset shape:", df.shape)
print(df.head())

# 2. Identify target and features
# Assuming dataset has a column named 'diagnosis' with values 'B' (benign) and 'M' (malignant)
# Adjust if your column name is different
target_col = 'diagnosis'

# Encode target (B=0, M=1)
le = LabelEncoder()
y = le.fit_transform(df[target_col])

# Drop non-numeric columns (like 'id' or 'diagnosis')
X = df.drop(columns=[target_col, 'id'], errors='ignore')

# Keep only numeric data
X = X.select_dtypes(include=[np.number])

# For visualization, take first 2 numeric features
X_vis = X.iloc[:, :2].values

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_vis, y, test_size=0.2, random_state=42)

# 4. Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Train SVM (Linear kernel)
linear_svm = SVC(kernel='linear', C=1)
linear_svm.fit(X_train, y_train)
y_pred_linear = linear_svm.predict(X_test)

# 6. Train SVM (RBF kernel)
rbf_svm = SVC(kernel='rbf', C=1, gamma='scale')
rbf_svm.fit(X_train, y_train)
y_pred_rbf = rbf_svm.predict(X_test)

# 7. Hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.1, 0.01],
    'kernel': ['rbf']
}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("\nBest Parameters:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)

# 8. Cross-validation score for linear SVM
cv_scores_linear = cross_val_score(linear_svm, X_train, y_train, cv=5)
print("\nCross-validation scores (Linear SVM):", cv_scores_linear)
print("Mean CV score:", cv_scores_linear.mean())

# 9. Evaluation
print("\n--- Linear Kernel ---")
print("Accuracy:", accuracy_score(y_test, y_pred_linear))
print(classification_report(y_test, y_pred_linear))

print("\n--- RBF Kernel ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rbf))
print(classification_report(y_test, y_pred_rbf))

# 10. Visualization function
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.show()

# 11. Plot decision boundaries
plot_decision_boundary(linear_svm, X_train, y_train, "SVM Linear Kernel Decision Boundary")
plot_decision_boundary(rbf_svm, X_train, y_train, "SVM RBF Kernel Decision Boundary")
