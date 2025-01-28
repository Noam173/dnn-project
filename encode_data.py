#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 22:35:43 2025

@author: noam
"""
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
def encoder(path):
    """
    

    Parameters
    ----------
    path : string
        path to the location of the dataset file.

    Returns
    -------
    string
        return the new location of the encoded dataset.

    """
    
    encoder=LabelEncoder()
    
    path=os.path.expanduser(path)
    
    data=pd.read_csv(path, dtype={'label': int})
    
    data = data[data['label'] != -1]
            
    if 'object' in data.dtypes.values:
        data=data.drop(['sha256', 'md5', 'appeared', 'entry', 'Unnamed: 0'], axis=1)
        
        categorial=data.select_dtypes(include=['object'])
        numerical=data.select_dtypes(exclude=['object'])

        del data

        for x in categorial.columns:
            categorial[x] = encoder.fit_transform(categorial[x])


        mixed=pd.concat([numerical,categorial], axis=1)
        
        path = os.path.splitext(path)[0]
        new_path=f'{path}_clean.csv'
        mixed.to_csv(new_path, index=False)
        return new_path;
    else:
        return path


