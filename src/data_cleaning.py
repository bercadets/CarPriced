import pandas as pd
import numpy as np

def simplify_transmission(val):
    if pd.isna(val):
        return 'Unknown'
    val = str(val).upper()  # uppercase everything so 'manual' and 'Manual' both match
    if 'CVT' in val or 'VARIABLE' in val or 'SINGLE' in val:
        return 'CVT'
    elif 'MANUAL' in val or 'M/T' in val or ' MT' in val:
        return 'Manual'
    elif 'AUTO' in val or 'A/T' in val or ' AT' in val:
        return 'Automatic'
    else:
        return 'Unknown'

# Load
path = "C:/Users/bercadets/.cache/kagglehub/datasets/taeefnajib/used-car-price-prediction-dataset/versions/1"
df = pd.read_csv(path + "/used_cars.csv")
print(f"Loaded {len(df)} cars")

# --- CLEAN PRICE ---
# Strip the $ and commas, then convert to number
df['price'] = df['price'].str.replace('$', '', regex=False)
df['price'] = df['price'].str.replace(',', '', regex=False)
df['price'] = pd.to_numeric(df['price'])

# --- CLEAN MILEAGE ---
df['milage'] = df['milage'].str.replace(' mi.', '', regex=False)
df['milage'] = df['milage'].str.replace(',', '', regex=False)
df['milage'] = pd.to_numeric(df['milage'])

# Verify
print("\nPrice sample:")
print(df['price'].head())
print("\nMileage sample:")
print(df['milage'].head())
print("\nTypes now:")
print(df[['price', 'milage']].dtypes)

#Clean that engine

# --- EXTRACT FROM ENGINE COLUMN ---

# Extract horsepower (e.g. "300.0HP" → 300.0)
df['horsepower'] = df['engine'].str.extract(r'(\d+\.?\d*)HP').astype(float)

# Extract engine size in liters (e.g. "3.7L" → 3.7)
df['liters'] = df['engine'].str.extract(r'(\d+\.?\d*)\s*[Ll](?:iter)?').astype(float)



# Extract cylinder count (e.g. "V6 Cylinder" → 6)
df['cylinders'] = df['engine'].str.extract(r'(\d+) Cylinder').astype(float)

# --- FILL MISSING ENGINE VALUES WITH MEDIAN ---
df['horsepower'] = df['horsepower'].fillna(df['horsepower'].median())
df['liters'] = df['liters'].fillna(df['liters'].median())
df['cylinders'] = df['cylinders'].fillna(df['cylinders'].median())

# Verify no more missing
print("\nMissing after filling:")
print(df[['horsepower', 'liters', 'cylinders']].isnull().sum())

# Check results
print("\nEngine extraction sample:")
print(df[['engine', 'horsepower', 'liters', 'cylinders']].head(10))

# Check how many NaNs we got
print("\nMissing after extraction:")
print(df[['horsepower', 'liters', 'cylinders']].isnull().sum())

# --- CLEAN FUEL TYPE ---
# Replace junk values with NaN first
df['fuel_type'] = df['fuel_type'].replace(['–', 'not supported'], np.nan)
# Fill missing with most common fuel type (gasoline)
df['fuel_type'] = df['fuel_type'].fillna(df['fuel_type'].mode()[0])

# --- CLEAN ACCIDENT ---
# Convert to binary: 1 = had accident, 0 = none reported
df['accident'] = df['accident'].map({
    'At least 1 accident or damage reported': 1,
    'None reported': 0
})
# Fill remaining NaN with 0 (assume no accident if not reported)
df['accident'] = df['accident'].fillna(0)

# --- CLEAN TITLE ---
# Yes = 1, NaN = 0
df['clean_title'] = df['clean_title'].map({'Yes': 1})
df['clean_title'] = df['clean_title'].fillna(0)

# Verify
print(df[['fuel_type', 'accident', 'clean_title']].head(10))
print("\nMissing values now:")
print(df[['fuel_type', 'accident', 'clean_title']].isnull().sum())

print("brand unique:", df['brand'].nunique())
print("model unique:", df['model'].nunique())
df['transmission'] = df['transmission'].apply(simplify_transmission)
print(df['transmission'].value_counts())
print("ext_col unique:", df['ext_col'].nunique())
print("int_col unique:", df['int_col'].nunique())

#dropping useless things/ used things
df = df.drop(['model', 'ext_col', 'int_col', 'engine'], axis=1)

# pd.get_dummies converts each category to its own 0/1 column
df = pd.get_dummies(df, columns=['brand', 'fuel_type', 'transmission'])

# --- REMOVE PRICE OUTLIERS ---
before = len(df)
df = df[(df['price'] >= 3000) & (df['price'] <= 200000)]
after = len(df)
print(f"Removed {before - after} outlier cars")
print(f"Remaining: {after} cars")
print(f"\nNew price range: ${df['price'].min():,} - ${df['price'].max():,}")

# Final check
print("Final shape:", df.shape)
print("\nFinal columns:")
print(df.columns.tolist())
print("\nMissing values:")
print(df.isnull().sum().sum(), "total missing")
print("\nFirst 3 rows:")
print(df.head(3))

df.to_csv('C:/Users/bercadets/Desktop/AI-Proj/2nd_try/data/clean_cars.csv', index=False)
print("Saved clean_cars.csv!")