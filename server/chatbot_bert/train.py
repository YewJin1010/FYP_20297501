import keras
import tensorflow_hub as hub

def build_smallBERT_CNN_classifier_model():
    text_input = keras.layers.Input(shape=(), dtype= keras.string, name='text')  # Fix dtype
    preprocessing_layer = hub.KerasLayer(bert_preprocessor, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(bert_encoder_model, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = sequence_output = outputs["sequence_output"]

    net = keras.layers.Conv1D(32, 2, activation='relu')(net)
    net = keras.layers.Conv1D(64, 2, activation='relu')(net)
    net = keras.layers.MaxPooling1D(pool_size=2)(net)
    
    net = keras.layers.GlobalMaxPooling1D()(net)  # Changed to GlobalMaxPooling1D
    
    net = keras.layers.Dense(512, activation="relu")(net)
    net = keras.layers.Dropout(0.1)(net)
    
    num_classes =  # Add the number of classes here
    
    net = keras.layers.Dense(num_classes, activation="softmax", name='classifier')(net)
    
    return keras.Model(text_input, net)

loss = keras.losses.CategoricalCrossentropy(from_logits=False)
metrics = keras.metrics.CategoricalAccuracy()  # Fix metrics
epochs = 15
optimizer = keras.optimizers.Adam(1e-5)

# build the model
classifier_smallBERT_model_cnn = build_smallBERT_CNN_classifier_model()

# compile the model
classifier_smallBERT_model_cnn.compile(optimizer=optimizer,
                                       loss=loss,
                                       metrics=[metrics])  # Fix metrics

class MyCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs['val_categorical_accuracy'] >= 0.94:
            print("\nValidation accuracy reached 94%!")
            self.model.stop_training = True

callback = MyCallback()
history = classifier_smallBERT_model_cnn.fit(x=X_train, y=y_train_one_hot,
                                             validation_data=(X_val, y_val_one_hot),
                                             batch_size=32,
                                             epochs=epochs,
                                             callbacks=[callback])
