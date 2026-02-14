import pandas as pd
import os

# Find the CSV file in data folder
data_folder = 'data'
csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

if csv_files:
    csv_path = os.path.join(data_folder, csv_files[0])
    print(f"Reading: {csv_path}")
    
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Basic info
    print("\n🔍 First 5 rows:")
    print(df.head())
    
    print("\n📊 Dataset shape:", df.shape)
    print("Columns:", list(df.columns))
    
    print("\n📋 Data types:")
    print(df.dtypes)
else:
    print("No CSV file found in data folder")

# Check correlation between numeric features and price
numeric_cols = ['Year', 'Engine Size', 'Mileage']
for col in numeric_cols:
    correlation = df[col].corr(df['Price'])
    print(f"Correlation between {col} and Price: {correlation:.3f}")