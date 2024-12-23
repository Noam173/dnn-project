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

   
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    x_train, y_train = tensor(x_train), tensor(y_train)
    x_val, y_val = tensor(x_val), tensor(y_val)
    

