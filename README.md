#H1Car Price Prediction AI Project
My first machine learning project - built on Feb14 because I was curious how AI works and also bored.

## H2Overview
I built this project to learn how machine learning actually works. It predicts car prices using features like brand, year, mileage, and engine size. Did it work well? No, It was terrible hahaha. Did I learn? Yup. Was it fun? Yes, so cool to see it analyze alot of data in a short amount of time.

What I Learned
Downloading data - Used kagglehub to grab a dataset (more on that below)
Exploring data - Actually LOOKING at data instead of just throwing it into a model
Cleaning data - Found 250 completely empty rows and yeeted them out
Preparing for ML - Learned that AI only understands numbers, not words like "Tesla" or "BMW"
Building a model - Trained a Random Forest model (fancy name, but it's just pattern finding)
Evaluating results - Figured out how to tell if a model is actually good or just guessing
Critical thinking - The most important lesson: sometimes data is just... bad

DataSet
import kagglehub
path = kagglehub.dataset_download("nalisha/car-price-prediction-dataset")
Shoutout to nalisha on Kaggle for the dataset!
Size: 2,500 cars with features like Brand, Year, Mileage, Engine Size, Transmission, Condition, and Price(some are empty)

Model Performance:

First Attempt (With "Car ID" included)
Average Error: ~$23,500

R² Score: -0.073

What went wrong? The AI decided that "Car ID" was the #1 most important factor for price. That's literally just a row number, imagine buying a car because it's #47 instead of #48. This taught me that AI is actually kind of dumb... it'll find patterns everywhere, even in meaningless data. It's my job to guide it.

Second Attempt (Removed Car ID)
Average Error: ~$24,000

R² Score: -0.103

Wait, it got WORSE?! Yep. Removing Car ID made things slightly worse, which was confusing at first.


Why such poor performance?

After actually analyzing the data (instead of just blaming my code), I found:


Relationship	        Correlation 	      What It Means
Year vs Price	        -0.035	            Basically zero connection
Mileage vs Price	    -0.010	            Also zero
Engine Size vs Price	-0.013	            You guessed it - zero

Summary:In this dataset, a 2023 BMW with 10,000 miles could be cheap OR expensive completely randomly. There's no real pattern to learn!

Conclusion: The dataset appears to be artificially generated, with random prices that don't follow real-world patterns. Even the best machine learning model can't find patterns that don't exist! (I just used this dataset because it was the most recent one I found on Kaggle.)

Key Takeaways
Machine learning isn't magic - it can only find patterns that actually exist in data

AI is dumb - It thought "Car ID" (literally just 1,2,3,4...) was important for pricing cars. It'll grab onto ANYTHING that looks like a pattern. We have to think for it.

Garbage in = garbage out - No matter how fancy your model is, bad data = bad results.

Look at your data first - I should've checked correlations before building everything. Would've saved time.My bad chat.

"Failure" - The model didn't work, but I learned WAY more than if I'd just copied working code.

Technologies Used
Python 3
pandas (data manipulation)
scikit-learn (machine learning)
matplotlib (visualization)
kagglehub (dataset download)
joblib (model saving)


About This Project
This is my first time building anything with AI. I literally learned by asking an AI to teach me (ironic, right?). I have no idea if I did everything "correctly" - there's probably a million things I could optimize or do better.

But that's kinda the point? I wanted to understand AI by actually building something, not just watching tutorials. The model didn't work well, but I honestly learned more from debugging than I would've from a perfect first try.

If you're reading this and thinking "this could be better" - you're probably right! This was me on Feb 14, 2026, bored on a Saturday, curious about AI, and just... building something.

Why Car Prices?
Honestly? I googled "cool machine learning projects for beginners" and car price prediction came up. Plus I've been getting into cars lately, so it seemed interesting. Sometimes the best reason to build something is just "why not?"
