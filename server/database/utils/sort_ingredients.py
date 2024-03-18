import json
import pandas as pd

# Load the data from cake.json
with open('server/database/unique_ingredients.json', 'r') as json_file:
    data = json.load(json_file)

def create_dataframe(primary, secondary, tertiary):
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
    
    return df

def sort_data(ingredients): 
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
    
    df = create_dataframe(primary, secondary, tertiary)

    # Write DataFrame to CSV file
    df.to_csv('server/database/sorted_ingredients.csv', index=False)

    return df

def clean_data(df):
  
    # Remove numbers and special characters
    df['Primary'] = df['Primary'].str.replace(r'[^a-zA-Z\s]', '')
    df['Secondary'] = df['Secondary'].str.replace(r'[^a-zA-Z\s]', '')
    df['Tertiary'] = df['Tertiary'].str.replace(r'[^a-zA-Z\s]', '')

    # Remove measurements
    measurements = ['stick', 'sticks', 'plus', 'cup', 'cups', 'teaspoon', 'teaspoons', 'tablespoon', 'tablespoons', 'tbsp', 'tsp', 'tbs', 'oz', 'ounce', 'ounces', 'pound', 'pounds', 'lb', 'g', 'gram', 'grams', 'kg', 'kilogram', 'kilograms', 'ml', 'millilitre', 'millilitres', 'l', 'litre', 'litres', 'pint', 'pints', 'quart', 'quarts', 'gallon', 'gallons', 'pt', 'qt', 'gal', 'gals', 'fl', 'fluid', 'fluids', 'oz', 'ounce', 'ounces', 'lb', 'pound', 'pounds', 'g', 'gram', 'grams', 'kg', 'kilogram', 'kilograms', 'ml', 'millilitre', 'millilitres', 'l', 'litre', 'litres', 'pint', 'pints', 'quart', 'quarts', 'gallon', 'gallons', 'pt', 'qt', 'gal', 'gals', 'fl', 'fluid', 'fluids', 'oz', 'ounce', 'ounces', 'lb', 'pound', 'pounds', 'g', 'gram', 'grams', 'kg', 'kilogram', 'kilograms', 'ml', 'millilitre', 'millilitres', 'l', 'litre', 'litres', 'pint', 'pints', 'quart', 'quarts', 'gallon', 'gallons', 'pt', 'qt', 'gal', 'gals', 'fl', 'fluid', 'fluids', 'oz', 'ounce', 'ounces', 'lb', 'pound', 'pounds', 'g', 'gram', 'grams', 'kg', 'kilogram', 'kilograms', 'ml', 'millilitre', 'millilitres', 'l', 'litre', 'litres', 'pint', 'pints', 'quart', 'quarts', 'gallon', 'gallons', 'pt', 'qt', 'gal', 'gals', 'fl', 'fluid', 'fluids', 'oz', 'ounce', 'ounces', 'lb']
    df['Primary'] = df['Primary'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (measurements)]))
    df['Secondary'] = df['Secondary'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (measurements)]))
    df['Tertiary'] = df['Tertiary'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (measurements)]))
    
    # Remove leading and trailing whitespaces
    df['Primary'] = df['Primary'].str.strip()
    df['Secondary'] = df['Secondary'].str.strip()
    df['Tertiary'] = df['Tertiary'].str.strip()

    # Remove duplicates
    primary = []
    secondary = []
    tertiary = []

    # Iterate over Primary column
    for ingredient in df['Primary']:
        if ingredient not in primary:
            primary.append(ingredient)
            # Remove ingredient from Secondary and Tertiary columns
            df['Secondary'] = df['Secondary'].replace(ingredient, '')
            df['Tertiary'] = df['Tertiary'].replace(ingredient, '')

    # Iterate over Secondary column
    for ingredient in df['Secondary']:
        if ingredient not in tertiary:
            secondary.append(ingredient)
            # Remove ingredient from Tertiary column
            df['Tertiary'] = df['Tertiary'].replace(ingredient, '')

    # Iterate over Tertiary column
    for ingredient in df['Tertiary']:
        tertiary.append(ingredient)

    # Remove empty strings
    primary = list(filter(None, primary))
    secondary = list(filter(None, secondary))
    tertiary = list(filter(None, tertiary))

    df = create_dataframe(primary, secondary, tertiary)
    
    # Write the cleaned DataFrame to a new CSV file
    df.to_csv('server/database/cleaned_ingredients.csv', index=False)
    return df

df = sort_data(data)
df = clean_data(df)