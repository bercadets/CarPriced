# clean_data.py

# Import the tools we need
import pandas as pd
import numpy as np

# Step 1: Load the data
print("📂 Loading data...")
df = pd.read_csv('data/car_price_prediction_with_missing.csv')
print(f"Initial shape: {df.shape}")

# Step 2: Remove the completely empty rows
# This finds rows where ALL columns are empty and removes them
df = df.dropna(how='all')
print(f"After removing completely empty rows: {df.shape}")

# Step 3: See what we're working with now
print("\n🔍 Missing values AFTER removing empty rows:")
print(df.isnull().sum())