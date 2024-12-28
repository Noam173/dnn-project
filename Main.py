# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import Data_Manipolation as dp
import Create_model as model


def main():
    path=dp.create_directory()
    file='~/train_features.csv'
    dp.Split_Train(file)
    model.create_model(32, 20)
    

    
if __name__=='__main__':
    main()
