import pandas as pd
import joblib

# Load the saved model
model = joblib.load('C:/Users/bercadets/Desktop/AI-Proj/2nd_try/data/car_price_model.pkl')

# Load clean data just to get the column names
df = pd.read_csv('C:/Users/bercadets/Desktop/AI-Proj/2nd_try/data/clean_cars.csv')
feature_columns = df.drop('price', axis=1).columns

# --- INPUT YOUR CAR ---
brand = input("Brand (e.g. Toyota, Ford, BMW): ").strip().title()
year = int(input("Year (e.g. 2018): "))
mileage = int(input("Mileage (e.g. 50000): "))
accident = int(input("Accident history? (1 = yes, 0 = no): "))
clean_title = int(input("Clean title? (1 = yes, 0 = no): "))
horsepower = float(input("Horsepower (e.g. 200): "))
liters = float(input("Engine size in liters (e.g. 2.5): "))
cylinders = float(input("Cylinders (e.g. 4): "))
fuel_type = input("Fuel type (Gasoline/Hybrid/Diesel/E85 Flex Fuel/Plug-In Hybrid): ").strip().title()
transmission = input("Transmission (Automatic/Manual/CVT/Unknown): ").strip().title()

# --- BUILD INPUT ROW ---
# Start with all zeros
input_data = pd.DataFrame([0] * len(feature_columns), index=feature_columns).T

# Fill in numeric values
input_data['model_year'] = year
input_data['milage'] = mileage
input_data['accident'] = accident
input_data['clean_title'] = clean_title
input_data['horsepower'] = horsepower
input_data['liters'] = liters
input_data['cylinders'] = cylinders

# Fill in one-hot encoded columns
brand_col = f'brand_{brand}'
fuel_col = f'fuel_type_{fuel_type}'
trans_col = f'transmission_{transmission}'

if brand_col in input_data.columns:
    input_data[brand_col] = 1
else:
    print(f"Warning: '{brand}' not recognized, treating as unknown brand")

if fuel_col in input_data.columns:
    input_data[fuel_col] = 1

if trans_col in input_data.columns:
    input_data[trans_col] = 1

# --- PREDICT ---
predicted_price = model.predict(input_data)[0]

# Our model's MAE was ~$7,050 so we use that as our margin
margin = 7050

low = predicted_price - margin
high = predicted_price + margin

print(f"\nEstimated Price Range: ${low:,.0f} - ${high:,.0f}")
print(f"Most likely around:    ${predicted_price:,.0f}")