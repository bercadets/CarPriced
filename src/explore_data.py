import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd


path = kagglehub.dataset_download("taeefnajib/used-car-price-prediction-dataset")
df = pd.read_csv(path + "/used_cars.csv")

# 1. Shape
print("Shape:", df.shape)

# 2. Column names and types
print("\nColumns and types:")
print(df.dtypes)

# 3. First few rows
print("\nFirst 5 rows:")
print(df.head())

# 4. Missing values
print("\nMissing values:")
print(df.isnull().sum())

# 5. Price statistics
print("\nPrice stats:")
print(df['price'].describe())  # might be 'Price' - check the column name!
print("Price examples:")
print(df['price'].head(10))

print("\nMileage examples:")
print(df['milage'].head(10))

print("\nEngine examples:")
print(df['engine'].head(10))

print("\nAccident unique values:")
print(df['accident'].unique())

print("\nClean title unique values:")
print(df['clean_title'].unique())

print("\nFuel type unique values:")
print(df['fuel_type'].unique())