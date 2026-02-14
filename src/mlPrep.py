import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the clean data
df = pd.read_csv('data/car_price_prediction_with_missing.csv')
df = df.dropna(how='all')
print(f"Loaded {len(df)} cars")

df = df.drop('Car ID', axis=1)  # Remove useless column

X = df.drop('Price', axis=1)  # Everything EXCEPT price
y = df['Price']  # Only price

print("\n Features (X):", list(X.columns))
print(" Target (y): Price")

# Convert text columns to numbers
print("\n Converting text to numbers...")

# Find which columns are text (object type)
text_columns = X.select_dtypes(include=['object']).columns
print("Text columns to convert:", list(text_columns))

# Convert each text column to numbers
label_encoders = {}
for col in text_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    print(f"   Converted {col}")


print("\n Data after conversion:")
print(X.head())

# training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n Data Split:")
print(f"  Training: {len(X_train)} cars (80%)")
print(f"  Testing: {len(X_test)} cars (20%)")

# Save the prepared data
X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)


print("\n Prepared data saved to data folder!")
