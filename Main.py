# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import Data_Manipulation as dp
import Create_model as model
import plot as plt
import encode_data as encode
import Reset_data as reset


def main():
    reset
    
    path = dp.create_directory()
    file='~/train_features_11.csv'
    file=encode.encoder(file)
    dp.Split_Train(file)
    hist,batch_size,num_epoch = model.create_model(32, 20)
    plt.plot_training_history(hist,batch_size,num_epoch)
    
    
if __name__=='__main__':
    main()
