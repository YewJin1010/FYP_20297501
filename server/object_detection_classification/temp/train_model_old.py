import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, Model, load_model
from keras import layers, optimizers
from keras.applications import ResNet50, MobileNetV2
from keras.applications.resnet50 import preprocess_input
from keras.callbacks import ModelCheckpoint, EarlyStopping, History, ReduceLROnPlateau
import keras.backend as K
from keras.layers import Input, MaxPool2D, MaxPooling2D, AveragePooling2D, add, Dense, Activation, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras import models, regularizers
from keras.models import Model
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight

def read_csv(file_path):
    return pd.read_csv(file_path)

def extract_columns(dataframe):
    return dataframe.keys().values.tolist()

def extract_subfolder(filename):
    parts = filename.split('_')
    if len(parts) == 2 and parts[-1].isdigit():
        return parts[0]
    else:
        return '_'.join(parts[:-1])

def create_data_generators(directory, csv_filename, columns):
    df = pd.read_csv(os.path.join(directory, csv_filename))
    df['filename'] = df['filename'].str.strip()
    df['full_path'] = df['filename'].apply(lambda x: os.path.join(directory, x))
    df['subfolder'] = df['filename'].apply(extract_subfolder)
    df['full_path'] = df.apply(lambda row: os.path.join(directory, row['subfolder'], row['filename']), axis=1)

    invalid_images = []

    for index, row in df.iterrows():
        if not os.path.exists(row['full_path']):
            invalid_images.append(row['filename'])
            print(f"Invalid image: {row['filename']}, Full path: {row['full_path']}")


    data_generator = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    data_flow = data_generator.flow_from_dataframe(
        dataframe=df[:],
        directory=directory,
        x_col="full_path",
        y_col=columns,
        batch_size=32,
        class_mode="raw",
        target_size=(224, 224)
    )
    
    return df, data_flow

def plot_images_per_class(class_counts, title):
    plt.figure(figsize=(10, 6))
    class_counts_numeric = pd.to_numeric(class_counts, errors='coerce')  # Convert to numeric, handle errors by converting to NaN
    class_counts_numeric.plot(kind='bar', color='skyblue')
    plt.title(title)
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45, ha='right')
    plt.show()

def create_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.compile(optimizers.RMSprop(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def create_resnet50_model(input_shape, num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False)
    for layer in base_model.layers:
        layer.trainable = False
    # Get base model output 
    base_model_ouput = base_model.output
    
    # Adding our own layer 
    x = GlobalAveragePooling2D()(base_model_ouput)
    # Adding fully connected layer
    x = Dense(512, activation='relu')(x)
    x = Dense(num_classes, activation='softmax', name='fcnew')(x)
    
    model = Model(inputs=base_model.input, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    """
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the weights of the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False
    
     # Add a new input layer to match the shape
    input_tensor = Input(shape=input_shape)
    x = base_model(input_tensor)
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)

    # Add more Convolutional layers
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)

    # Output layer
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=predictions)

    model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])
    """
    return model


def create_mobilenetv2_model(input_shape, num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    model = models.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='sigmoid'))
    model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_generator, valid_generator, epochs, callbacks=[]):
    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size

    print("Training generator samples:", train_generator.n)
    print("Validation generator samples:", valid_generator.n)

    lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=1e-7)

    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        validation_data=valid_generator,
        validation_steps=STEP_SIZE_VALID,
        epochs=epochs,
        callbacks= [lr_schedule] + callbacks,
        shuffle=False,
    )

    return model

def evaluate_model(model, test_generator):
    test_loss, test_acc = model.evaluate(test_generator)
    print("The test loss is: ", test_loss)
    print("The best accuracy is: ", test_acc * 100)

def save_model(model, filename, model_save_path):
    # Construct a unique filename with model_type and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_type}_{timestamp}.h5"
    filepath = os.path.join(model_save_path, filename)
    
    # Check if the file already exists
    count = 1
    while os.path.exists(filepath):
        filename = f"{model_type}_{timestamp}_{count}.h5"
        filepath = os.path.join(model_save_path, filename)
        count += 1
    
    # Save the model
    model.save(filepath)
    print(f"Model saved at: {filepath}")

    return model

# Paths
# C:/Users/yewji
# "c:/Users/miku/Documents/Yew Jin/
# Read CSVs
train_data = read_csv('C:/Users/miku/Documents/Yew Jin/FYP_20297501/server/object_detection/train/_classes.csv')
test_data = read_csv('C:/Users/miku/Documents/Yew Jin/FYP_20297501/server/object_detection/test/_classes.csv')
valid_data = read_csv('C:/Users/miku/Documents/Yew Jin/FYP_20297501/server/object_detection/valid/_classes.csv')

# Directory Paths
train_dir = "C:/Users/miku/Documents/Yew Jin/FYP_20297501/server/object_detection/train"
test_dir = "C:/Users/miku/Documents/Yew Jin/FYP_20297501/server/object_detection/test"
valid_dir = "C:/Users/miku/Documents/Yew Jin/FYP_20297501/server/object_detection/valid"

# Model Directory
model_save_path = "C:/Users/miku/Documents/Yew Jin/FYP_20297501/server/object_detection/models"
model_filename = "C:/Users/miku/Documents/Yew Jin/FYP_20297501/server/object_detection/models/resnet50.h5"

# Plot Path
plot_path = "C:/Users/miku/Documents/Yew Jin/FYP_20297501/server/object_detection/results"

# Result CSV
results_csv_path = "C:/Users/miku/Documents/Yew Jin/FYP_20297501/server/object_detection/results/results.csv"
test_csv_path = "C:/Users/miku/Documents/Yew Jin/FYP_20297501/server/object_detection/test/_classes.csv"     

# Training code
train_epochs = int(input("Enter the number of training epochs: "))
model_type = input("Enter the model type (cnn/resnet50/mobilenetv2): ")

# Extract Columns
train_columns = extract_columns(train_data)

# Remove filename column
columns = [col for col in train_columns if col != 'filename']

# Create Data Generators
train_df, train_generator = create_data_generators(train_dir, "_classes.csv", columns)
test_df, test_generator = create_data_generators(test_dir, "_classes.csv", columns)
valid_df, valid_generator = create_data_generators(valid_dir, "_classes.csv", columns)

# User input for selecting which plots to display
print("Select plots to display:")
print("1. Training Plot")
print("2. Validation Plot")
print("3. Testing Plot")
print("4. All Plots")
print("5. No Plot")
choice = input("Enter your choice (1/2/3/4/5): ")

# Display plots based on user choice
if choice == "1":
    plot_images_per_class(train_df[columns].sum(), 'Number of Images per Class for Training')
elif choice == "2":
    plot_images_per_class(valid_df[columns].sum(), 'Number of Images per Class for Validation')
elif choice == "3":
    plot_images_per_class(test_df[columns].sum(), 'Number of Images per Class for Testing')
elif choice == "4":
    plot_images_per_class(train_df[columns].sum(), 'Number of Images per Class for Training')
    plot_images_per_class(valid_df[columns].sum(), 'Number of Images per Class for Validation')
    plot_images_per_class(test_df[columns].sum(), 'Number of Images per Class for Testing')
elif choice == "5":
    print("No plots selected.")
else:
    print("Invalid choice. Please enter a valid option (1/2/3/4/5).")

# Model creation and training
if model_type.lower() == 'cnn':
    model = create_cnn_model(input_shape=(224, 224, 3), num_classes=len(columns))
elif model_type.lower() == 'resnet50':
    model = create_resnet50_model(input_shape=(224, 224, 3), num_classes=len(columns))
elif model_type.lower() == 'mobilenetv2':
    model = create_mobilenetv2_model(input_shape=(224, 224, 3), num_classes=len(columns))
else:
    print("Invalid model type. Supported types: cnn, resnet50, mobilenetv2.")
    exit()

# Display the model summary
model.summary()

callbacks = [
    ModelCheckpoint(f'best_{model_type}_model.h5', save_best_only=True),
    EarlyStopping(patience=3, restore_best_weights=True),
    History()
]

model = train_model(model, train_generator, valid_generator, train_epochs, callbacks)

model = save_model(model, f'{model_type}.h5', model_save_path)

history = callbacks[-1]

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy and Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Construct a unique filename for the plot
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plot_filename = f"training_validation_plot_{model_type}_{timestamp}.png"

# Save the plot
results_location = 'C:/Users/miku/Documents/Yew Jin/FYP_20297501/server/object_detection/results'
plt.savefig(os.path.join(results_location, plot_filename))
#plt.show()

evaluate_model(model, test_generator)


