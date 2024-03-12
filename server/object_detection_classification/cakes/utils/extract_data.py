import json
import pandas as pd

# Load the data from cake.json
with open('server/object_detection_classification/cakes/dataset/cake.json', 'r') as json_file:
    data = json.load(json_file)

def extract_data_to_json(criteria_list):
    combined_data = []

    # Iterate through each recipe
    for recipe in data:
        recipe_data = {}
        # Extract data for each criterion
        for criteria in criteria_list:
            recipe_data[criteria] = recipe.get(criteria, "")
        combined_data.append(recipe_data)

    # Output file
    output_file = 'server/object_detection_classification/cakes/data.json'

    # Write the combined data to the file
    with open(output_file, 'w') as file:
        json.dump(combined_data, file, indent=4)
    
    print(f'Data extracted and saved to {output_file}')


def extract_data_csv(criteria_list):
    # Initialize a dictionary to store data for each criterion
    extracted_data = {criteria: [] for criteria in criteria_list}

    # Iterate through each recipe in the JSON data
    for recipe in data:
        # Extract data for each criterion
        for criteria in criteria_list:
            extracted_data[criteria].append(recipe.get(criteria, ""))
    
    # Create a DataFrame from the extracted data
    df = pd.DataFrame(extracted_data)
    
    csv_file = 'server/object_detection_classification/cakes/data.csv'
    # Write DataFrame to CSV file
    df.to_csv(csv_file, index=False)
    
    print(f'Data extracted and saved to {csv_file}')

  
# User input for the criteria
print("Enter criteria separated by commas (e.g., ingredients, url, title, directions):")
criteria_input = input("Enter criteria: ")

print("Enter format (json or csv):")
format_input = input("Enter format: ")

# Extract individual criteria from user input
criteria_list = [criteria.strip() for criteria in criteria_input.split(',')]

if format_input == 'json':
    # Call the function to extract data based on the provided criteria
    extract_data_to_json(criteria_list)

elif format_input == 'csv':
    # Call the function to extract data based on the provided criteria
    extract_data_csv(criteria_list)

else:
    print("Invalid format. Please enter 'json' or 'csv'")