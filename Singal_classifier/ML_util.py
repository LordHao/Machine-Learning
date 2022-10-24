import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

def plot_loss_tf(history):
    fig,ax = plt.subplots(1,2, figsize = (20,8))
    ax[0].plot(history.history['loss'], marker = 'x',label='loss')
    ax[0].set_xlabel('Epoch',size =25)
    ax[0].set_ylabel('loss ',size =25)
    ax[0].legend(fontsize =25)
    
    ax[1].plot(history.history['accuracy'], marker = 'x',label='accuracy')
    ax[1].set_xlabel('Epoch',size =25)
    ax[1].set_ylabel('accuracy',size =25)
    ax[1].legend(fontsize =25)
    plt.show()

def bc_predict(model,x,y):
    model_predict = lambda X: np.around(tf.nn.sigmoid(model.predict(X)).numpy())
    yhat = model_predict(x).reshape(-1)
    return yhat 

def bc_eva(model,x,y):
    y_hat = bc_predict(model,x,y)
    tp = len(y_hat[y_hat==1])
    tn = len(y_hat[y_hat==0])

    itp = np.where(y_hat == 1)[0]
    itn = np.where(y_hat == 0)[0]
    
    fp = len(np.where(y[itp]==0)[0])
    fn = len(np.where(y[itn]==1)[0])
    
    accuracy = np.mean([y_hat==y])
    print('accuracy = ', accuracy)
    return accuracy



def DF(df):
    pd.set_option('display.max_columns', None)
    display(df)
    