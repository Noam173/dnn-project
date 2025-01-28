import os
import pandas as pd
from sklearn.model_selection import train_test_split


def create_directory():
    '''
    

    Returns
    -------
    string.
        path to a folder which will be used as a placeholder for the train, test and validation of the dataset.

    '''
    path='/home/noam/dataSet'
    if not (os.path.exists(path)):
        print("Making directory...")
        os.mkdir(path)
        os.mkdir(f'{path}/Train')
        os.mkdir(f'{path}/Validation')
        os.mkdir(f'{path}/Test')
    return path;


def Split_Train(train_path):
    '''
    Parameters
    ----------
    train_path : string.
        the path to the dataset file to prepare it for splitting.

    Returns
    -------
    None.

    '''
    
    train_path=os.path.expanduser(train_path)
    path=pd.read_csv(train_path)
    
    x = path.drop('label', axis=1) 
    y = path['label']
    del path
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3,shuffle=True, stratify=y.to_numpy())
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.33,shuffle=True,stratify=y_temp.to_numpy())
    del x_temp,y_temp
    
    if not os.path.exists(f'{create_directory()}/Train/x_train.csv' or f'{create_directory()}/Train/y_train.csv'):
        x_train.to_csv(f'{create_directory()}/Train/x_train.csv',index=False)
        y_train.to_csv(f'{create_directory()}/Train/y_train.csv',index=False)
    if not os.path.exists(f'{create_directory()}/Validation/x_val.csv' or f'{create_directory()}/Validation/y_val.csv'):
        x_val.to_csv(f'{create_directory()}/Validation/x_val.csv',index=False)
        y_val.to_csv(f'{create_directory()}/Validation/y_val.csv',index=False)
    if not os.path.exists(f'{create_directory()}/Test/x_test.csv' or f'{create_directory()}/Test/y_test.csv'):
        x_test.to_csv(f'{create_directory()}/Test/x_test.csv',index=False)
        y_test.to_csv(f'{create_directory()}/Test/y_test.csv',index=False)
    print(f'{create_directory()}, suDir: Train, Validation, Test')
    
    
    
def Get_train():
    '''
    

    Returns
    -------
    x_train : DataFrame
        read's the x_train's values from the csv file.
    y_train : DataFrame
        read's the y_train's values from the csv file.

    '''
    dir = create_directory()
    x_train = pd.read_csv(f'{dir}/Train/x_train.csv')
    y_train = pd.read_csv(f'{dir}/Train/y_train.csv')
    return x_train, y_train

def Get_val():
    '''
    

    Returns
    -------
    x_train : DataFrame
        read's the x_val's values from the csv file.
    y_train : DataFrame
        read's the y_val's values from the csv file.

    '''
    dir = create_directory()
    x_val = pd.read_csv(f'{dir}/Validation/x_val.csv')
    y_val = pd.read_csv(f'{dir}/Validation/y_val.csv')
    return x_val, y_val

def Get_test():
    '''
    

    Returns
    -------
    x_train : DataFrame
        read's the x_test's values from the csv file.
    y_train : DataFrame
        read's the y_test's values from the csv file.

    '''
    dir = create_directory()
    x_test = pd.read_csv(f'{dir}/Test/x_test.csv')
    y_test = pd.read_csv(f'{dir}/Test/y_test.csv')
    return x_test, y_test

    
    
    
    
    
    

