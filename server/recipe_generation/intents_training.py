import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
from keras.preprocessing.sequence import pad_sequences

nltk_data_path = 'C:/Users/yewji/FYP_20297501/server/recipe_generation/nltk_data'
model_path = 'C:/Users/yewji/FYP_20297501/server/recipe_generation/model'
# Set the NLTK data path
nltk.data.path.append(nltk_data_path)

words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('C:/Users/yewji/FYP_20297501/server/recipe_generation/intents.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:

        # take each word and tokenize it
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # adding documents
        documents.append((w, intent['tag']))

        # adding classes to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print (len(documents), "documents")

print (len(classes), "classes", classes)

print (len(words), "unique lemmatized words", words)

words_path = os.path.join(nltk_data_path, 'texts.pkl')
classes_path = os.path.join(nltk_data_path, 'labels.pkl')

pickle.dump(words, open(words_path, 'wb'))
pickle.dump(classes, open(classes_path, 'wb'))

# initializing training data
training = []
output_empty = [0] * len(classes)
for doc in documents:
    # initializing bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Get the maximum length of bags
max_length = max(len(bag) for bag, _ in training)

# Pad or truncate bags to a fixed length
training_padded = [(pad_sequences([bag], maxlen=max_length)[0], output_row) for bag, output_row in training]

# Shuffle the padded training data
random.shuffle(training_padded)

# Convert the training data to a NumPy array
training = np.array(training_padded, dtype=object)

# create train and test lists. X - patterns, Y - intents
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Training data created")


# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save(os.path.join(model_path, 'chatbot_model.h5'), hist)

print("model created")