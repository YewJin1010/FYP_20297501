# Import the necessary modules
import pandas as pd
import json

json_data = json.load(open('C:/Users/yewji/FYP_20297501/server/recipe_recommendation/bert/datasets/cake_recipes.json'))

# Convert to csv
df = pd.DataFrame(json_data)
df.to_csv('C:/Users/yewji/FYP_20297501/server/recipe_recommendation/bert/datasets/cake_recipes.csv', index=False)
print(df.head())

# Load the data from a CSV file
df = pd.read_csv('C:/Users/yewji/FYP_20297501/server/recipe_recommendation/bert/datasets/cake_recipes.csv')
print("number of rows before dropping duplicates:", df.shape[0])

# Drop duplicate rows
df.drop_duplicates(subset=['title'], inplace=True)
print("number of rows after dropping duplicates:", df.shape[0])

# Write the modified DataFrame to the csv file
df.to_csv('C:/Users/yewji/FYP_20297501/server/recipe_recommendation/bert/datasets/cake_recipes.csv', index=True, index_label='id')

# Create a new DataFrame with only the 'title' and 'ingredients' columns
new_df = df[['title', 'ingredients']]

# Save the new DataFrame to a new CSV file
new_df.to_csv('C:/Users/yewji/FYP_20297501/server/recipe_recommendation/bert/datasets/title_ingredient.csv', index=False)
