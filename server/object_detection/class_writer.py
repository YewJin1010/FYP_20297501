import pandas as pd
import shutil
import os

def drop_duplicates_in_column():
    csv_path = 'C:/Users/miku/Documents/Yew Jin/cakes/valid/_classes.csv'

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Drop duplicate values in the specified column
    df = df.drop_duplicates(subset=['filename'])

    # Write the changes back to the CSV file
    df.to_csv(csv_path, index=False)

    # Display the modified DataFrame
    print(df)

def update_new_classes(class_mapping):
    old_csv_path = 'C:/Users/miku/Documents/Yew Jin/datasets/ingredientsv2i/valid/_classes.csv'
    new_csv_path = 'C:/Users/miku/Documents/Yew Jin/cakes/valid/_classes.csv'
    
    # Read the old and new CSV files
    df_old = pd.read_csv(old_csv_path)
    df_new = pd.read_csv(new_csv_path)

    # Map old class names to new class names
    for old_class, new_class in class_mapping.items():
        # Check if the old class exists in the old CSV columns
        if old_class in df_old.columns:
            if new_class not in df_new.columns:
                df_new[new_class]=0

            # Update the corresponding column in the new CSV based on the old CSV
            #df_new[new_class] = df_new['filename'].isin(df_old[df_old[old_class] == 1]['filename']).astype(int)
            df_new.loc[df_new['filename'].isin(df_old[df_old[old_class] == 1]['filename']), new_class] = 1

    # Write the changes back to the new CSV file
    df_new.to_csv(new_csv_path, index=False)

print("1. Drop duplicates in column")
print("2. Update new classes")

selection = input("Enter your selection: ")

if selection == "1":
    drop_duplicates_in_column()

elif selection == "2":
    class_mapping = {
    ' Wheat Flour': 'Flour',
    ' Sugar': 'Sugar'
    # Add more mappings as needed
}
    update_new_classes( class_mapping)
