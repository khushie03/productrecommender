import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import numpy as np
height = int(input("Enter your height in centimeters : "))
weight = float(input("enter your weight in kgs : "))
BMI = (weight) / ((height/100)**2)
print("your bmi is ", BMI)
if BMI < 18.5:
    print("Underweight")
if BMI > 18.5 and BMI <24.9 :
    print("Normal Weight")
if BMI > 25 and BMI <29.9:
    print("Overweight")
if BMI > 30 and BMI <34.9:
    print("Obesity Class 1")
if BMI > 35 and BMI < 39.9:
    print("Obesity Class 2")
if BMI >= 40:
    print("Obesity Class 3 (Morbid Obesity)")
# Sample user profile defined in the code
user_profile = {
    "user_id": 6,
    "health_conditions": "depression, anxiety , weight loss",
    "user_Description": "non-smoker, drinking , vegetarian",
    "gender": "Female",
    "age": 28
}



# Load product data from the JSON file (without user_profile)
with open(r"C:\Users\Khushi Pandey\Desktop\numpy\supplements.json", 'r') as json_file:
    data = json.load(json_file)

# Extract product data and benefits
product_data = pd.DataFrame(data.get('data', []))
product_to_benefits = {}
for product, benefits in data.get('benefits', {}).items():
    product_to_benefits[product] = ', '.join(benefits)

product_data['product_description'] = product_data['product_name'].map(product_to_benefits)

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
product_descriptions = product_data['product_description'].fillna('')
tfidf_matrix = tfidf_vectorizer.fit_transform(product_descriptions)

user_health_conditions = user_profile['health_conditions']
user_vector = tfidf_vectorizer.transform([user_health_conditions])

item_scores = np.dot(user_vector, tfidf_matrix.T).toarray()

softmax_scores = np.exp(item_scores) / np.sum(np.exp(item_scores), axis=1, keepdims=True)

num_recommendations = 5
recommended_product_indices = np.argsort(softmax_scores, axis=1)[:, -num_recommendations:]

recommended_products = []
for i in range(len(recommended_product_indices[0])):
    product_index = recommended_product_indices[0][i]
    recommended_products.append(product_data['product_name'].iloc[product_index])

print("Recommended Products:")
for product in recommended_products:
    print(product)

