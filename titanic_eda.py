# Task 2: Exploratory Data Analysis (EDA) - Titanic Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# ---------------- Load Dataset (Direct from GitHub) ----------------
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# ---------------- Basic Info ----------------
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nData Types:\n", df.dtypes)

# ---------------- Summary Statistics ----------------
print("\nSummary Statistics:\n", df.describe())
print("\nMissing Values:\n", df.isnull().sum())

# ---------------- Histograms for Numeric Features ----------------
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
df[numeric_cols].hist(figsize=(12, 8), bins=20)
plt.suptitle("Histograms of Numeric Features", fontsize=16)
plt.show()

# ---------------- Boxplots for Numeric Features (Dynamic Grid) ----------------
plt.figure(figsize=(12, 6))
rows = math.ceil(len(numeric_cols) / 3)  # 3 plots per row
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(rows, 3, i)
    sns.boxplot(y=df[col])
    plt.title(f"Boxplot - {col}")
plt.tight_layout()
plt.show()

# ---------------- Correlation Matrix ----------------
plt.figure(figsize=(10, 6))
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap", fontsize=16)
plt.show()

# ---------------- Optimized Pairplot (Only Key Columns) ----------------
selected_cols = ["Age", "Fare", "Pclass", "Survived"]
sns.pairplot(df[selected_cols].dropna())
plt.suptitle("Pairplot of Selected Features", y=1.02)
plt.show()

# ---------------- Categorical vs Target ----------------
plt.figure(figsize=(6, 4))
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x="Pclass", hue="Survived", data=df)
plt.title("Survival by Passenger Class")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival by Gender")
plt.show()

# ---------------- Observations ----------------
print("\nBasic Observations:")
print("1. Higher survival rate for females compared to males.")
print("2. Higher survival rate for passengers in 1st class.")
print("3. Age distribution shows many passengers between 20–40 years.")
print("4. Fare distribution is skewed, with some very high outliers.")
