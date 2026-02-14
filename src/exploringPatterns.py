# explore_patterns.py
import pandas as pd
import matplotlib.pyplot as plt

# Load the clean data
df = pd.read_csv('data/car_price_prediction_with_missing.csv')
# Remove those empty rows (just to be safe)
df = df.dropna(how='all')

print("📊 DATA EXPLORATION")
print("=" * 50)
print(f"Total cars: {len(df)}")
print(f"Features: {list(df.columns)}")

# Let's look at price distribution
print("\n💰 PRICE STATISTICS")
print(df['Price'].describe())

# Question 1: Which brand has the highest average price?
brand_prices = df.groupby('Brand')['Price'].mean().sort_values(ascending=False)
print("\n🏭 AVERAGE PRICE BY BRAND:")
print(brand_prices)

# Question 2: How does year affect price?
print("\n📅 PRICE BY YEAR:")
year_prices = df.groupby('Year')['Price'].mean()
print(year_prices)

# Question 3: How does mileage affect price?
print("\n⛽ MILLAGE vs PRICE (first 10 cars):")
print(df[['Mileage', 'Price']].head(10))