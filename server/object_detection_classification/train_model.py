import pandas as pd 
import numpy as np 
from sklearn.preprocessing import OneHotEncoder 
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.layers import Input, MaxPool2D, MaxPooling2D, AveragePooling2D, add, Dense, Activation, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
import matplotlib.pyplot as plt
from keras.callbacks import History
from keras.applications.imagenet_utils import preprocess_input
from keras.callbacks import ModelCheckpoint, EarlyStopping, History, ReduceLROnPlateau
import glob, os, shutil
from keras.optimizers import Adam
import time

def extract_columns(dataframe):
    return dataframe.keys().values.tolist()

def extract_subfolder(filename):
    parts = filename.split('_')
    if len(parts) == 2 and parts[-1].isdigit():
        return parts[0]
    else:
        return '_'.join(parts[:-1])

def create_data_generators(data_generator, directory, df, classes):
    df['filename'] = df['filename'].str.strip()
    df['full_path'] = df['filename'].apply(lambda x: os.path.join(directory, x))
    df['subfolder'] = df['filename'].apply(extract_subfolder)

    pd.set_option('display.max_colwidth', None)  # Disable truncation
    df['full_path'] = df.apply(lambda row: os.path.join(directory, row['subfolder'], row['filename']), axis=1)
    df['full_path'] = df['full_path'].str.replace('\\', '/')

    invalid_images = []
    for index, row in df.iterrows():
        if not os.path.exists(row['full_path']):
            invalid_images.append(row['filename'])
            print(f"Invalid image: {row['filename']}, Full path: {row['full_path']}")
    
    data_flow = data_generator.flow_from_dataframe(
        dataframe=df,
        directory='',
        x_col="full_path",
        y_col=classes,
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
    plt.savefig(f'server/object_detection_classification/dataset/{title}.png')
    plt.show()

def plot_images_train_valid(train_counts, valid_counts, title):
    plt.figure(figsize=(10, 6))
    train_counts_numeric = pd.to_numeric(train_counts, errors='coerce')
    valid_counts_numeric = pd.to_numeric(valid_counts, errors='coerce')
    
    # Plot training data
    plt.bar(train_counts_numeric.index, train_counts_numeric.values, color='skyblue', label='Training')
    
    # Plot validation data
    plt.bar(valid_counts_numeric.index, valid_counts_numeric.values, color='yellow', label='Validation')
    
    plt.title(title)
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.savefig(f'server/object_detection_classification/dataset/{title}.png')
    plt.show()
  
# Create ResNet50 model
def create_resnet50_model(input_shape, num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
    # Get base model output 
    base_model_ouput = base_model.output

    x = GlobalAveragePooling2D()(base_model_ouput)
    # Adding fully connected layer
    x = Dense(512, activation='relu')(x)
    x = Dense(num_classes, activation='softmax', name='fcnew')(x)
    
    model = Model(inputs=base_model.input, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    
    return model

def train_model(model, train_generator, valid_generator, epochs):
    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size

    print("Training generator samples:", train_generator.n)
    print("Validation generator samples:", valid_generator.n)

    model.fit(
        train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        validation_data=valid_generator,
        validation_steps=STEP_SIZE_VALID,
        epochs=epochs,
        shuffle=False,
        callbacks=callbacks,
        verbose=1
    )
    return model

def evaluate_model(model, test_generator):
    test_loss, test_acc = model.evaluate(test_generator)
    print("The test loss is: ", test_loss)
    print("The best accuracy is: ", test_acc * 100)

# Read CSVs
train_data = pd.read_csv('server/object_detection_classification/dataset/train/_classes.csv')
valid_data = pd.read_csv('server/object_detection_classification/dataset/valid/_classes.csv')

# Directory Paths
train_dir = "server/object_detection_classification/dataset/train"
valid_dir = "server/object_detection_classification/dataset/valid"

# Model Save Path
model_save_path = "server/object_detection_classification/trained_models"
checkpoint_save_path = "server/object_detection_classification/trained_models/checkpoints"

# Results Path
plot_path = "server/object_detection_classification/results/training_plots"

# User input for number of training epochs
train_epochs = int(input("Enter the number of training epochs: "))

# Extract Columns
train_columns = extract_columns(train_data)

# Remove filename column
classes = [col for col in train_columns if col != 'filename']

# Create Data Generators
train_data_gen = ImageDataGenerator(preprocessing_function = preprocess_input, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
valid_data_gen = ImageDataGenerator(preprocessing_function = preprocess_input)
train_df, train_generator = create_data_generators(train_data_gen, train_dir, train_data, classes)
valid_df, valid_generator = create_data_generators(valid_data_gen, valid_dir, valid_data, classes)

# User input for selecting which plots to display
print("Display train & valid plot:")
choice = input("Enter your choice (y/n): ")
choice.lower()
# Display plots based on user choice
if choice == "y":
    plot_images_per_class(train_df[classes].sum(), 'Number of Images per Class for Training')
    plot_images_per_class(valid_df[classes].sum(), 'Number of Images per Class for Validation')
    plot_images_train_valid(train_df[classes].sum(), valid_df[classes].sum(), 'Number of Images per Class')

elif choice == "n":
    print("No plots selected.")
else:
    print("Invalid choice. Please enter a valid option (y/n).")

model = create_resnet50_model(input_shape=(224, 224, 3), num_classes=len(classes))

model.summary()

file_name = "resnet50"

callbacks = [
    ModelCheckpoint(
        filepath=os.path.join(checkpoint_save_path, f'base_{file_name}.h5'),
        save_best_only=True
    ),
    History()
]
# Train model
model = train_model(model, train_generator, valid_generator, train_epochs)

# Save model
now = time.strftime("%d-%m-%Y_%H-%M-%S")
model_path = os.path.join(model_save_path, f'{file_name}_{now}_model.h5')
model.save(model_path)

history = callbacks[-1]

# Plot training and validation loss
plt.figure()  # Create a new figure for the loss plot
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(plot_path, f'{file_name}_{now}_loss.png'))

# Plot training and validation accuracy
plt.figure()  
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(plot_path, f'{file_name}_{now}_accuracy.png'))

            