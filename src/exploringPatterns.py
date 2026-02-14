import pandas as pd
import matplotlib.pyplot as plt

# Load the clean data
df = pd.read_csv('data/car_price_prediction_with_missing.csv')

df = df.dropna(how='all')

print("📊 DATA EXPLORATION")
print("=" * 50)
print(f"Total cars: {len(df)}")
print(f"Features: {list(df.columns)}")


print("\n PRICE STATISTICS")
print(df['Price'].describe())


brand_prices = df.groupby('Brand')['Price'].mean().sort_values(ascending=False)
print("\n AVERAGE PRICE BY BRAND:")
print(brand_prices)


print("\n PRICE BY YEAR:")
year_prices = df.groupby('Year')['Price'].mean()
print(year_prices)


print("\n MILLAGE vs PRICE (first 10 cars):")

print(df[['Mileage', 'Price']].head(10))
