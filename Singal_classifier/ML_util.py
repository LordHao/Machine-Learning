import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import itertools

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 25)
    plt.colorbar()
    plt.tick_marks = np.arange(len(classes))
    plt.xticks(plt.tick_marks, classes, rotation=45)
    plt.yticks(plt.tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', size = 20)
    plt.xlabel('Predicted label', size = 20)

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

def bc_predict(model,x):
    model_predict = lambda X: np.around(tf.nn.sigmoid(model.predict(X)).numpy())
    yhat = model_predict(x).reshape(-1)
    return yhat 

def bc_eva(model,x,y):
    
    y_hat = bc_predict(model,x)

    idp = np.where(y_hat == 1)[0]
    idn = np.where(y_hat == 0)[0]
    
    tp = len(np.where(y[idp]==1)[0])
    tn = len(np.where(y[idn]==0)[0])
    
    fp = len(np.where(y[idp]==0)[0])
    fn = len(np.where(y[idn]==1)[0])
    
    recall = tp/(tp+fn); precision = tp/(tp+fp)
    Fs = 2*recall*precision/(recall+precision)
    
    accuracy = np.mean([y_hat==y])
    
    print('model: %s'%model.name)
    print('accuracy = ', accuracy)
    print('recall = ', recall)
    print('precision = ', precision)
    print('F-score = ', Fs)
    
    return accuracy,recall,precision,Fs



def DF(df):
    pd.set_option('display.max_columns', None)
    display(df)