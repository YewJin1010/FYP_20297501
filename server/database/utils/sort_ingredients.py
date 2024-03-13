import json
import pandas as pd

# Load the data from cake.json
with open('server/database/unique_ingredients.json', 'r') as json_file:
    data = json.load(json_file)

def categorise_ingredients(ingredients): 
    primary = []
    secondary = []
    tertiary = []

    for ingredient, count in ingredients:
        if count > 100:
            primary.append(ingredient)
        elif count > 1:
            secondary.append(ingredient)
        else:
            tertiary.append(ingredient)
    
    return primary, secondary, tertiary

def write_to_csv(primary, secondary, tertiary):
    max_length = max(len(primary), len(secondary), len(tertiary))
    
    # Fill arrays with empty strings to ensure equal length
    primary += [''] * (max_length - len(primary))
    secondary += [''] * (max_length - len(secondary))
    tertiary += [''] * (max_length - len(tertiary))
    
    # Create a DataFrame from the extracted data
    df = pd.DataFrame({
        'Primary': primary,
        'Secondary': secondary,
        'Tertiary': tertiary
    })
    
    # Write DataFrame to CSV file
    df.to_csv('server/database/sorted_ingredients.csv', index=False)
    

primary, secondary, tertiary = categorise_ingredients(data)
write_to_csv(primary, secondary, tertiary)