from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import pandas as pd

# Load dataset
df = pd.read_csv('server/recipe_recommendation/t5/csv/recipes_t5.csv')
print(df.info())

# Define tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Define training parameters
batch_size = 4
num_epochs = 3
learning_rate = 1e-4

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Fine-tuning loop
for epoch in range(num_epochs):
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        
        input_text = batch['ingredients'].tolist()
        target_text = batch['title_directions'].tolist()
        try: 
            input_encodings = tokenizer(input_text, padding=True, truncation=True, return_tensors='pt')
            target_encodings = tokenizer(target_text, padding=True, truncation=True, return_tensors='pt')
        except Exception as e: 
            print("Exception: ", e)
            continue
        
        input_ids = input_encodings['input_ids']
        attention_mask = input_encodings['attention_mask']
        labels = target_encodings['input_ids']
        
        optimizer.zero_grad()
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(df)}], Loss: {loss.item()}")

# Save the fine-tuned model and tokenizer
model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")
