# Car Price Prediction AI Project
Built on Feb 14, 2026 out of curiosity and boredom. Then rebuilt on March 7, 2026 because the first one was terrible — and actually learned why.

---

## Overview
I built this project to learn how machine learning actually works. It predicts used car prices using features like brand, year, mileage, engine size, horsepower, and accident history.

**Version 1 (Feb 14):** Terrible results. R² of -0.073. Dataset was fake.

**Version 2 (March 7):** R² of 0.842 with XGBoost. Real data, proper cleaning, feature engineering, and model comparison.

---

## The Dataset

### Version 1 (Bad Data)
```python
path = kagglehub.dataset_download("nalisha/car-price-prediction-dataset")
```
2,500 cars. Prices were randomly generated — no real patterns. A 2023 BMW with 10,000 miles could be the same price as a 2005 Honda with 200,000 miles. Garbage in, garbage out.

### Version 2 (Good Data)
```python
path = kagglehub.dataset_download("taeefnajib/used-car-price-prediction-dataset")
```
4,009 real US used car listings with brand, model year, mileage, engine specs, transmission, accident history, and price.

---

## What Changed: Version 1 vs Version 2

| Thing | V1 | V2 |
|-------|----|----|
| Dataset | Fake/random prices | Real US listings |
| Cleaning | Dropped empty rows | Full pipeline |
| Features | Basic | Extracted HP, liters, cylinders from engine text |
| Encoding | LabelEncoder (wrong) | One-Hot Encoding (correct) |
| Models | Random Forest only | Linear Regression, Random Forest, XGBoost |
| Outlier removal | None | Removed cars < $3k and > $200k |
| Best R² | -0.103 | 0.842 |
| Best MAE | ~$24,000 | ~$7,050 |

---

## Data Cleaning Pipeline

The raw data had several messy columns that needed fixing before the model could use them:

**Price and Mileage** — stored as strings like `"$10,300"` and `"51,000 mi."`. Stripped symbols and converted to integers.

**Engine column** — stored as messy text like `"300.0HP 3.7L V6 Cylinder Engine Gasoline Fuel"`. Used regex to extract three separate numeric features:
- Horsepower (`300.0`)
- Engine size in liters (`3.7`)
- Cylinder count (`6`)

**Transmission** — 62 unique messy values simplified down to 4 clean categories: Automatic, Manual, CVT, Unknown.

**Accident and Clean Title** — converted text categories to binary 0/1 values.

**Fuel Type** — replaced junk values (`"–"`, `"not supported"`) with the most common value (Gasoline).

**Outlier removal** — dropped cars priced below $3,000 or above $200,000. This single step improved R² from 0.119 to 0.842.

**One-Hot Encoding** — converted brand (57 unique), fuel type (6), and transmission (4) into binary columns. Final dataset: 4,009 rows × 74 columns, 0 missing values.

---

## Model Comparison

Trained three models and compared them on the same test set (785 cars):

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | $11,824 | $19,658 | 0.630 |
| Random Forest | $8,107 | $14,593 | 0.796 |
| **XGBoost** | **$7,132** | **$12,828** | **0.842** |

XGBoost won. It builds trees sequentially, each one learning from the previous one's mistakes — which works really well for tabular data like this.

### Hyperparameter Tuning

Used GridSearchCV to try 54 combinations of XGBoost settings across 5-fold cross validation (270 total training runs). Best parameters found:

```
learning_rate: 0.1
max_depth: 5
n_estimators: 300
subsample: 0.8
```

Tuned R² (cross-validation): **0.861**

The tuned model's test R² was 0.825 vs default's 0.842 — tuning didn't help much here, which is normal for smaller datasets. More data would help more than tuning.

---

## Real World Test

Tested the model against a real listing on CarGurus:

### 2021 Hyundai Sonata Limited — Listed at $16,827

![2021 Hyundai Sonata on CarGurus](hyundai_sonata_test.jpg)

![Prediction result](prediction_result.png)

**Model predicted: $18,076** — off by about $1,249. That's within our MAE range and honestly pretty solid for a used car price prediction.

---

## Known Limitations

**US market only.** The model was trained on US listings. PH market prices follow different patterns due to import taxes, local demand, and different depreciation rates. Multiplying by exchange rate gives a rough estimate but not accurate PH predictions.

**2024+ cars are rare** in the training data so predictions for very new cars are less reliable.

**No model column.** Dropped because it had 1,898 unique values. This means the model can't distinguish a BMW 1 Series from a BMW M5 — it just sees "BMW." Brand tier feature engineering would help here.

---

## Key Takeaways

**Garbage in = garbage out.** Version 1 failed entirely because the data was fake. No model can learn patterns that don't exist.

**Outlier removal was the biggest win.** One filter (remove prices outside $3k-$200k) took R² from 0.119 to 0.842. Clean data beats fancy models.

**AI is dumb in a specific way.** It finds patterns everywhere — even in meaningless data like Car ID numbers. Guiding what it sees is the engineer's job.

**LabelEncoder was wrong.** Encoding `Audi=0, BMW=1, Ford=2` tells the model Ford is "more" than Audi. One-hot encoding fixes this.

**More data beats more tuning.** Hyperparameter tuning gave marginal improvement. A dataset with 50k+ cars would likely improve results more than any tuning.

**Domain matters.** A model trained on US prices predicts US prices. Applying it to PH market introduces a fundamental mismatch that math alone can't fix.

---

## Technologies Used
- Python 3
- pandas — data manipulation
- scikit-learn — Linear Regression, Random Forest, preprocessing, GridSearchCV
- xgboost — XGBoost regressor
- kagglehub — dataset download
- joblib — model saving
- regex — engine feature extraction

---

## About This Project

First version was me on Feb 14, 2026, bored on a Valentine's Day Saturday, curious about AI. It failed completely and I learned exactly why.

Second version was built with proper guidance — real data, real cleaning, real feature engineering, and actual understanding of each step. The model went from R² of -0.103 to 0.842.

The goal was never to build a perfect model. It was to understand how machine learning actually works by building something real, breaking it, and figuring out why.

If you're reading this thinking "this could be better" — you're probably right. But Feb 14 me knew nothing about any of this, and March 7 me built something that actually works. That's the point.
