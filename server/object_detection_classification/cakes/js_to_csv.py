import pandas as pd
import json 

# Load the JSON file
with open('server/object_detection_classification/cakes/dataset/cakes.json') as f:
    data = json.load(f)


# Convert the JSON file to a DataFrame
df = pd.DataFrame(data)
df.to_csv('server/object_detection_classification/cakes/dataset/cakes.csv', index=True, index_label='id')
