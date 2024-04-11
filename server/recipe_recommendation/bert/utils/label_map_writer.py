import pandas as pd

# Read the titles from the CSV file
df = pd.read_csv('recipe_recommendation/bert/dataset/cleaned_recipes_dataset.csv')
titles = df['title'].tolist()

# Create a label map
label_map = {}
label_number = 1
for title in titles:
    # Check if the title contains a colon
    if ':' not in title:
        if title not in label_map:
            label_map[title] = label_number
            label_number += 1

# Define the path to save the text file
txt_file_path = 'recipe_recommendation/bert/dataset/label_map.txt'

# Write the label map to a text file
with open(txt_file_path, 'w') as f:
    for title, label in label_map.items():
        f.write(f"{title}: {label}\n")
