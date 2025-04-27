# 1. Import Libraries
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 2. Load the California Housing Dataset
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Optional: Convert to DataFrame for easy viewing
df = pd.DataFrame(X, columns=housing.feature_names)
df['MEDIAN_HOUSE_VALUE'] = y

# 3. Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict on Test Set
y_pred = model.predict(X_test)

# 6. Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# 7. Plot Actual vs Predicted Prices
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Prices (in $100,000)")
plt.ylabel("Predicted Prices (in $100,000)")
plt.title("Actual vs Predicted House Prices (California)")
plt.show()
