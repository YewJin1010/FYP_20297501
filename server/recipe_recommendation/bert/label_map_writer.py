import pandas as pd

# Read the titles from the CSV file
df = pd.read_csv('C:/Users/yewji/FYP_20297501/server/recipe_recommendation/bert/datasets/cleaned_title_ingredient.csv')
titles = df['title'].tolist()

# Create a label map
label_map = {}
label_number = 1
for title in titles:
    if title not in label_map:
        label_map[title] = label_number
        label_number += 1

# Define the path to save the text file
txt_file_path = 'C:/Users/yewji/FYP_20297501/server/recipe_recommendation/bert/label_map.txt'

# Write the label map to a text file
with open(txt_file_path, 'w') as f:
    for title, label in label_map.items():
        f.write(f"{title}: {label}\n")