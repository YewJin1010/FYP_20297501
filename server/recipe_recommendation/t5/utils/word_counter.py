import pandas as pd
import matplotlib.pyplot as plt

def count_data(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Create a new DataFrame to store word counts
    word_counts = pd.DataFrame()

    # Iterate over each column in the DataFrame
    for column in df.columns:
        # Check if the column contains NaN values
        if df[column].isna().any():
            # Replace NaN values with empty strings
            df[column] = df[column].fillna('')

        # Split the text in each cell into words and count the number of words
        word_counts[column] = df[column].astype(str).str.split().apply(len)

    # Plot word distribution for each column
    for column in word_counts.columns:
        plt.figure(figsize=(8, 6))
        plt.hist(word_counts[column], bins=30, color='skyblue', edgecolor='black')
        plt.title(f'Word Distribution for {column}')
        plt.xlabel('Number of Words')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

# Path to the CSV file
#csv_path = 'server/recipe_recommendation/t5/dataset/recipes_t5.csv'
csv_path = 'server/recipe_recommendation/t5/dataset_backup/new_data.csv'

# Call the function to count data
count_data(csv_path)
