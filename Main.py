# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import Data_Manipulation as dp
import Create_model as model
import plot as plt


def main():
    path = dp.create_directory()
    file='~/train_features.csv'
    dp.Split_Train(file)
    hist = model.create_model(batch_size=32, num_epoch=20)
    plt.plot_training_history(hist)

    
if __name__=='__main__':
    main()
