import os
import pandas as pd
import json 

base_directory = 'server/database/cakes/index/c/'
md_file_path = 'server/database/dataset/cake.md'
js_file_path = 'server/database/dataset/cake.json'
csv_file_path = 'server/database/dataset/cake.csv'

def md_to_js(md_file_path, base_directory, js_file_path):
    # Open the Markdown file
    with open(md_file_path, 'r', encoding='utf-8') as file:
        markdown_content = file.read()

    # Parse the Markdown content to extract links
    lines = markdown_content.split('\n')
    links = [line.strip().split('](')[-1][:-1] for line in lines if line.strip().startswith('* [')]

    # Open destination file
    with open(js_file_path, 'w', encoding='utf-8') as dest_file:
        # Create JSON array
        dest_file.write("[")  

        # Append each linked JSON file
        for i, link in enumerate(links):
            # Construct the absolute path
            file_path = os.path.normpath(os.path.join(base_directory, link)) 

            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as source_file:
                    recipe_content = source_file.read()
                    dest_file.write(recipe_content)
                    if i < len(links) - 1:
                        # Seperate recipes
                        dest_file.write(',')  
                    dest_file.write('\n')
                print(f"Appended: {file_path}")
            else:
                print(f"File not found: {file_path}")

        dest_file.write("]")  # End the JSON array
        print("Done")

def js_to_csv(js_file_path, csv_file_path):
    # Load the JSON file
    with open(js_file_path) as f:
        data = json.load(f)

    # Convert the JSON file to a DataFrame
    df = pd.DataFrame(data)
    df.to_csv(csv_file_path, index=True, index_label='id')

md_to_js(md_file_path, base_directory, js_file_path)
js_to_csv(js_file_path, csv_file_path)

