import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training(history):

    data=pd.DataFrame.from_dict(history)
    data['epochs']=data.index+1

    sns.lineplot(x='epochs', y='accuracy', data=data, label='Train accuracy')
    sns.lineplot(x='epochs', y='val_accuracy',  data=data, label='Val accuracy')
    plt.show()
    
    sns.lineplot(x='epochs', y='loss', data=data, label='Train loss')
    sns.lineplot(x='epochs', y='val_loss', data=data, label='Val loss')
    plt.show()