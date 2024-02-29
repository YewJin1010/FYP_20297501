intent_mapping = df.groupby('intent_encoding')['intent'].first().to_dict()

user_utterance = "what should i prepare before i go"
user_utterance = np.array([user_utterance])

# Make the prediction
predictions = loaded_model.predict(user_utterance)
predicted_intent_index = np.argmax(predictions)

# Map the predicted intent index to intent label and response
predicted_intent_label = intent_mapping[str(predicted_intent_index)]
predicted_response = df.loc[df['intent'] == predicted_intent_label, 'response'].iloc[0]

print("Predicted Intent:", predicted_intent_label)
print("VELA:", predicted_response)