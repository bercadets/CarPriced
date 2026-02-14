# prepare_for_ml.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the clean data
df = pd.read_csv('data/car_price_prediction_with_missing.csv')
df = df.dropna(how='all')
print(f"Loaded {len(df)} cars")

# Add this line right after loading the data
df = df.drop('Car ID', axis=1)  # Remove useless column

# Step 1: Separate features (X) and target (y)
X = df.drop('Price', axis=1)  # Everything EXCEPT price
y = df['Price']  # Only price (what we want to predict)

print("\n🔍 Features (X):", list(X.columns))
print("🎯 Target (y): Price")

# Step 2: Convert text columns to numbers
print("\n🔄 Converting text to numbers...")

# Find which columns are text (object type)
text_columns = X.select_dtypes(include=['object']).columns
print("Text columns to convert:", list(text_columns))

# Convert each text column to numbers
label_encoders = {}
for col in text_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    print(f"  ✅ Converted {col}")

# Step 3: Check the result
print("\n📊 Data after conversion:")
print(X.head())

# Step 4: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n📊 Data Split:")
print(f"  Training: {len(X_train)} cars (80%)")
print(f"  Testing: {len(X_test)} cars (20%)")

# Save the prepared data
X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)

print("\n✅ Prepared data saved to data folder!")