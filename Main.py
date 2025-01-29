# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import Data_Manipulation as dp
from Create_model import create_model as model
from plot import plot_training as plt
import encode_data as encode
import Reset_data as reset


def main():
    reset
    
    path = dp.create_directory()
    file='~/train_features.csv'
    file=encode.encoder(file)
    dp.Split_Train(file)
    
    hist, = model(128, 50)
    plt(hist)

    hist, = model(32, 50)
    plt(hist)
    
    
if __name__=='__main__':
    main()
