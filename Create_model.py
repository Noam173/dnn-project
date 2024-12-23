import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import Data_Manipolation as dp
from keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def create_model(batch_size, num_epoch):
    numpy = lambda np: np.to_numpy()
    tensor = lambda x: tf.convert_to_tensor(x, dtype=tf.float32)

    # Load data
    x_train, y_train = dp.Get_train()
    x_val, y_val = dp.Get_val()

    x_train, y_train = numpy(x_train), numpy(y_train)
    x_val, y_val = numpy(x_val), numpy(y_val)

    # Normalize the data (mean 0, std 1)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    x_train, y_train = tensor(x_train), tensor(y_train)
    x_val, y_val = tensor(x_val), tensor(y_val)
    
   
    # Early stopping callback
   
    
    model = Sequential()
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01), input_shape=(x_train.shape[1],))) 
    model.add(Dropout(0.3))
    
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01))) 

    model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.01))) 

    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01))) 
    model.add(Dropout(0.2))

    model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))) 
    model.compile(Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])

    print("\n\nModel summary:")
    print(model.summary())  # Print model summary
    
    # Training the model with early stopping
    hist = model.fit(x_train, y_train,
                     batch_size=batch_size,
                     epochs=num_epoch,
                     validation_data=(x_val, y_val))
    
    plot_training_history(hist)

def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss during Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting the accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy during Training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()