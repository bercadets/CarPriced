# build_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the prepared data
X_train = pd.read_csv('data/X_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').squeeze()  # Convert to 1D array
y_test = pd.read_csv('data/y_test.csv').squeeze()

print("🚗 CAR PRICE PREDICTION MODEL")
print("=" * 50)
print(f"Training data: {X_train.shape[0]} cars")
print(f"Testing data: {X_test.shape[0]} cars")

# Step 1: Create and train the model
print("\n Training Random Forest model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 2: Make predictions
print(" Making predictions...")
y_pred = model.predict(X_test)

# Step 3: Evaluate accuracy
print("\n  MODEL PERFORMANCE")
print("=" * 50)

# Mean Absolute Error - average error in dollars
mae = mean_absolute_error(y_test, y_pred)
print(f"  Average error: ${mae:,.2f}")
print(f"   (off by about ${mae:,.0f} on average)")

# R² Score - how well it explains price variation (0-1, higher is better)
r2 = r2_score(y_test, y_pred)
print(f"\n R² Score: {r2:.3f}")
print(f"   (1.0 = perfect, 0.0 = random guessing)")

# Step 4: Compare predictions vs actual
print("\n SAMPLE PREDICTIONS (first 10 test cars):")
comparison = pd.DataFrame({
    'Actual Price': y_test[:10].values,
    'Predicted Price': y_pred[:10].round(2),
    'Difference': (y_test[:10].values - y_pred[:10]).round(2)
})
print(comparison)

# Step 5: Feature importance - what matters most?
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔑 TOP 5 MOST IMPORTANT FEATURES:")
print(feature_importance.head(5))

# Step 6: Simple visualization
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Car Prices')
plt.tight_layout()
plt.show()

# Save the model
import joblib
joblib.dump(model, 'models/car_price_model.pkl')
print("\n✅ Model saved to 'models/car_price_model.pkl'")