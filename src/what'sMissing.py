import pandas as pd
import os

# Load the data
df = pd.read_csv('data/car_price_prediction_with_missing.csv')

print("🔍 MISSING DATA CHECK")
print("=" * 40)

# Check for missing values in each column
missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100

# Create a nice table
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing %': missing_percent
})

print(missing_df[missing_df['Missing Count'] > 0])

print("\n📊 Total missing values:", df.isnull().sum().sum())
print("Dataset shape:", df.shape)

# Quick stats of numerical columns
print("\n📈 Price Statistics:")
print(df['Price'].describe())