import pandas as pd

def count_classes(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Exclude the 'filename' column from counting
    class_columns = df.columns[1:]

    # Create a dictionary to store the count of each class
    class_count = {}

    # Calculate the count for each class
    for column in class_columns:
        class_count[column] = df[column].sum()

    return class_count

# Example usage
csv_file_path = 'C:/Users/miku/Documents/Yew Jin/cakes/train/_classes.csv'
result = count_classes(csv_file_path)

# Print the class counts
for class_name, count in result.items():
    print(f"{class_name}: {count}")