import os

md_file_path = 'server/object_detection_classification/cakes/index/c/cake.md'
base_directory = 'server/object_detection_classification/cakes/index/c/'
destination_file_path = 'server/object_detection_classification/cakes/recipes.json'

with open(md_file_path, 'r', encoding='utf-8') as file:
    markdown_content = file.read()

# Parse the Markdown content to extract links
lines = markdown_content.split('\n')
links = [line.strip().split('](')[-1][:-1] for line in lines if line.strip().startswith('* [')]

# Open destination file
with open(destination_file_path, 'w', encoding='utf-8') as dest_file:
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
