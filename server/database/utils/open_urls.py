import json
import webbrowser

# Load the data from baking.json
with open('server/database/dataset/cake.json', 'r') as json_file:
    data = json.load(json_file)

def open_url_by_range(start_index, end_index):
    # Open URLs within the specified range
    for index in range(start_index, min(end_index + 1, len(data))):
        url = data[index]['url']
        webbrowser.open(url)

    print(f"Opened URLs from index {start_index} to {min(end_index, len(data) - 1)}")

start_index = int(input("Enter start index: "))
end_index = int(input("Enter end index: "))

# Call the function to open URLs based on the provided range
open_url_by_range(start_index, end_index)

