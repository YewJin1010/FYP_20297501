import os

# Read the cake.md file
md_file_path = 'C:/Users/miku/Documents/Yew Jin/recipes/index/c/cake.md'  # Replace with the actual path

with open(md_file_path, 'r', encoding='utf-8') as file:
    markdown_content = file.read()

# Specify the directory to save downloaded JSON files
download_directory = 'downloaded_json'
os.makedirs(download_directory, exist_ok=True)

# Base directory where JSON files are stored
base_directory = 'C:/Users/miku/Documents/Yew Jin/recipes/index/c/'  # Replace with the actual base directory

# Destination file to append the recipes
destination_file_path = 'C:/Users/miku/Documents/Yew Jin/recipes/index/c/all_recipes.json'

# Parse the Markdown content to extract links
lines = markdown_content.split('\n')
links = [line.strip().split('](')[-1][:-1] for line in lines if line.strip().startswith('* [')]

# Open the destination file
with open(destination_file_path, 'w', encoding='utf-8') as dest_file:
    dest_file.write("[")  # Start the JSON array

    # Append each linked JSON file
    for i, link in enumerate(links):
        # Construct the absolute path and remove `../../`
        file_path = os.path.normpath(os.path.join(base_directory, link))

        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as source_file:
                recipe_content = source_file.read()
                dest_file.write(recipe_content)
                if i < len(links) - 1:
                    dest_file.write(',')  # Add a comma between recipes
                dest_file.write('\n')
            print(f"File appended: {file_path}")
        else:
            print(f"File not found: {file_path}")

    dest_file.write("]")  # End the JSON array
    print("All recipes appended.")
