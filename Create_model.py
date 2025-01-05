import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import Data_Manipulation as dp
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def create_model(batch_size, num_epoch):
    '''
    this function retrives and convert the already created train, val, and test 
    files to tensor (for the model), than creates the model and uses the user's inputs for refrence.

    Parameters
    ----------
    batch_size : int, x^2.
        how many data samples to take in each batch, determent the length of each epoch.
       
    num_epoch : int.
        how many epochs to train the model on.    

    Returns
    -------
    None.

    '''
    numpy = lambda np: np.to_numpy()
    tensor = lambda x: tf.convert_to_tensor(x, dtype=tf.float32)

    x_train, y_train = dp.Get_train()
    x_val, y_val = dp.Get_val()
    x_test, y_test = dp.Get_test()

    x_train, y_train = numpy(x_train), numpy(y_train)
    x_val, y_val = numpy(x_val), numpy(y_val)
    x_test, y_test = numpy(x_test), numpy(y_test)
   
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    x_train, y_train = tensor(x_train), tensor(y_train)
    x_val, y_val = tensor(x_val), tensor(y_val)
    x_test, y_test = tensor(x_test), tensor(y_test)
    
    early_stopping = EarlyStopping(monitor="val_loss",
                                   patience=2,         
                                   restore_best_weights=True,  
                                   verbose=1)
    
    model = Sequential()
    model.add(Dense(128, activation='relu',input_shape=(x_train.shape[1],)))

    model.add(Dense(64, activation='relu'))

    model.add(Dense(32, activation='relu'))
    

    model.add(Dropout(0.5))


    model.add(Dense(1, activation='sigmoid'))


    model.compile(Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    print("\n\nModel summary:")
    print(model.summary())  
    

    hist = model.fit(x_train, y_train,
                     batch_size=batch_size,
                     epochs=num_epoch,
                     validation_data=(x_val, y_val),
                     callbacks=[early_stopping])
    
    
    loss, accuracy=model.evaluate(x_test ,y_test)
    print(f"Test Accuracy: {accuracy:.2f}, {loss:.2f}")
    
    return hist.history;


