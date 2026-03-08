import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib

# --- LOAD CLEANED DATA ---
df = pd.read_csv('C:/Users/bercadets/Desktop/AI-Proj/2nd_try/data/clean_cars.csv')

# --- SEPARATE FEATURES AND TARGET ---
X = df.drop('price', axis=1)
y = df['price']

# --- SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training: {len(X_train)} cars")
print(f"Testing: {len(X_test)} cars")

# --- DEFINE MODELS ---
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
}

# --- TRAIN AND EVALUATE EACH MODEL ---
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{name}")
    print(f"  MAE:  ${mae:,.0f}")
    print(f"  RMSE: ${rmse:,.0f}")
    print(f"  R²:   {r2:.3f}")

# --- TUNE XGBOOST ---
print("\n" + "="*50)
print("TUNING XGBOOST...")
print("="*50)

param_grid = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.05, 0.1, 0.3],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0]
}

print("Total combinations:", 
    len(param_grid['n_estimators']) * 
    len(param_grid['learning_rate']) * 
    len(param_grid['max_depth']) * 
    len(param_grid['subsample'])
)

xgb = XGBRegressor(random_state=42)

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\nBest parameters:", grid_search.best_params_)
print("Best R² (cross-validation):", grid_search.best_score_.round(3))

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nTuned XGBoost on test set:")
print(f"  MAE:  ${mae:,.0f}")
print(f"  RMSE: ${rmse:,.0f}")
print(f"  R²:   {r2:.3f}")

joblib.dump(best_model, 'C:/Users/bercadets/Desktop/AI-Proj/2nd_try/data/car_price_model.pkl')
print("Model saved!")