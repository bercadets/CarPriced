import pandas as pd
import numpy as np

# Load the data
print("📂 Loading data...")
df = pd.read_csv('data/car_price_prediction_with_missing.csv')
print(f"Initial shape: {df.shape}")


# This finds rows where ALL columns are empty and removes them
df = df.dropna(how='all')
print(f"After removing completely empty rows: {df.shape}")


print("\n🔍 Missing values AFTER removing empty rows:")

print(df.isnull().sum())
