# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 09:34:55 2025

@author: Student
"""

import matplotlib.pyplot as plt

def plot_training_history(history):
    '''
    

    Parameters
    ----------
    history : object.
        the model's results after training.
    epochs : int.
        takes the epochs's from the create_model function.
    batch_size : int, x^2.
        takes the batch size's from the create_model function.

    Returns
    -------
    None.

    '''
    
    plt.figure(figsize=(12, 6))
    plt.suptitle(f'Number of epochs: {history['epochs']}, Batch size: {history['batch_size']}')
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss during Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()


    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy during Training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()


    plt.tight_layout()
    plt.show()

