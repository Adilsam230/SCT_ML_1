import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

try:
    df = pd.read_csv(r"C:\Users\adils\OneDrive\Documents\house_data.txt")
except FileNotFoundError:
    print("Error: 'housing_data.csv' not found. Make sure the file is in the same directory.")
    exit()
print("--- Data Loaded Successfully ---")

features = ['SquareFootage', 'Bedrooms', 'Bathrooms']
X = df[features]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
print("\n--- Model Training Complete ---")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"\nModel Coefficients: {model.coef_}")
print(f"Model Intercept: {model.intercept_:.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
new_house = pd.DataFrame({
    'SquareFootage': [500],
    'Bedrooms': [2],
    'Bathrooms': [1]
})

predicted_price = model.predict(new_house)

print("\n--- New House Prediction ---")
print(f"Details: {new_house.to_dict('records')[0]}")
print(f"Predicted Price: ${predicted_price[0]:,.2f}")