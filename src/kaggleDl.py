import kagglehub
import shutil
import os

# Download the dataset
path = kagglehub.dataset_download("nalisha/car-price-prediction-dataset")
print("Dataset downloaded to:", path)

# Find the CSV file
for file in os.listdir(path):
    if file.endswith('.csv'):
        csv_file = os.path.join(path, file)
        print("Found CSV:", csv_file)
        
        # Create data folder if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Copy to your data folder
        destination = os.path.join('data', file)
        shutil.copy2(csv_file, destination)
        print(f"✅ CSV copied to: {destination}")

# List what's in your data folder
print("\n📁 Files in your data folder:")
print(os.listdir('data'))