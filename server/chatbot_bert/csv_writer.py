import csv
import json

# Load intents from JSON file
with open('server/chatbot_bert/intents.json', 'r') as file:
    intents_data = json.load(file)

# Extract patterns and tags
patterns = []
tags = []
for intent in intents_data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

# Write patterns and tags to CSV file
csv_path = 'server/chatbot_bert/dataset.csv'
with open(csv_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['sentence', 'intent'])  # Write header
    for pattern, tag in zip(patterns, tags):
        writer.writerow([pattern, tag])

