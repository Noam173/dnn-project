# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import Data_Manipulation as dp
from Create_model import create_model as model
from plot import plot_training_history as plt
import encode_data as encode
import Reset_data as reset


def main():
    reset
    
    path = dp.create_directory()
    file='~/train_features.csv'
    file=encode.encoder(file)
    dp.Split_Train(file)
    x={128, 32}
    for num_epochs in x:
        hist,batch_size,num_epoch = model(num_epochs, 50)
        plt(hist,batch_size,num_epoch)
    
    
if __name__=='__main__':
    main()
