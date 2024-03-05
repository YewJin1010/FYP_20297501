import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def CNN_model():
    model = Sequential()
    model.add(Dense(units=4, activation='relu', input_dim= 66))  # Hidden layer with 4 neurons
    model.add(Dense(units=2832, activation='softmax'))  # Output layer with softmax activation
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(choice):
    csv_path = 'C:/Users/yewji/FYP_20297501/server/recipe_recommendation/neural_network/csv/dataset.csv'
    df = pd.read_csv(csv_path)

    X = df.drop(columns=['title']).values
    y = np.arange(len(df))

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    """
    # Initialize a Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    clf.fit(X_train, y_train)

    # Make predictions on the validation set
    y_pred = clf.predict(X_val)

    # Evaluate model performance
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy:.2f}")

    # Example: Predict the cuisine for a new recipe
    new_recipe = np.array([[1, 0, 1]])  # New recipe with ingredients
    predicted_cuisine = clf.predict(new_recipe)
    print(f"Predicted Cuisine: {predicted_cuisine[0]}")

    """
    """
    # number of rows
    print("Number of rows: ", df.shape[0])

    # Step 2: Preprocess Data
    # Separate features (ingredients) and labels (recipe titles)
    x = df.iloc[0:, 1:]
    y = df.iloc[:, 0]

    # Step 3: Split Data
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)


    # Label encode the labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_val = label_encoder.fit_transform(y_val)

    # Convert ingredients to a numpy array
    X_train = X_train.to_numpy()
    X_val = X_val.to_numpy()
    """

    if choice == 1:
        model = CNN_model()


    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')

    model.save('C:/Users/yewji/FYP_20297501/server/recipe_recommendation/neural_network/trained_models/model.h5')

    # Plot training and validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

print("1. CNN model")
choice = int(input("Select an option: "))
if choice == 1:
    train_model(choice)

