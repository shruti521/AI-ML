# Task 3: Linear Regression on Housing.csv

# 1️⃣ Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 2️⃣ Load dataset
df = pd.read_csv("Housing.csv")
print(df.head())
print(df.info())

# 3️⃣ Convert categorical columns to numeric using one-hot encoding
df_encoded = pd.get_dummies(df, drop_first=True)

# 4️⃣ Choose features (X) and target (y)
# Example: Predict 'price' based on other features
X = df_encoded.drop(columns=['price'])
y = df_encoded['price']

# 5️⃣ Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6️⃣ Fit Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 7️⃣ Predictions
y_pred = model.predict(X_test)

# 8️⃣ Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.2f}")

# 9️⃣ Plot: Predicted vs Actual
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.show()
