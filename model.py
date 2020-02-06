# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 11:18:40 2019

@author: thieu
"""
#import dependencies
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras import regularizers
from keras.optimizers import SGD
from keras.layers import Dense,Flatten, Dropout
from keras.layers import MaxPooling2D,Conv2D
#define loss/accuracy plot
def show_final_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].set_title('loss')
    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax[1].set_title('accuracy')
    ax[1].plot(history.epoch, history.history["accuracy"], label="Train accuracy")
    ax[1].plot(history.epoch, history.history["val_accuracy"], label="Validation accuracy")
    ax[0].legend()
    ax[1].legend()
    
#define function for loading data
def load_data(path):
    data = loadmat(path)
    return data['X'], data['y']

x_train, y_train = load_data('train_32x32.mat')
x_test, y_test = load_data('test_32x32.mat')
# Transpose the image arrays
x_train, y_train = x_train.transpose((3,0,1,2)), y_train[:,0]
x_test, y_test = x_test.transpose((3,0,1,2)), y_test[:,0]
print("Training Set", x_train.shape)
print("Test Set", x_test.shape)
#split tranining to train and validation
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.12, random_state=42)
#normalize data
#Calculate the mean on the training data
mean = np.mean(x_train, axis=0)
#Calculate the standard deviation on the training data
std = np.std(x_train, axis=0)
# Subtract it from all splits
train_norm = (x_train - mean) / std
test_norm = (x_test - mean)  / std
val_norm = (x_val - mean) / std
#encoding categorical data to fit cnn model
encoder = OneHotEncoder().fit(y_train.reshape(-1, 1))
#transform the label values 
y_train = encoder.transform(y_train.reshape(-1, 1)).toarray()
y_test = encoder.transform(y_test.reshape(-1, 1)).toarray()
y_val = encoder.transform(y_val.reshape(-1, 1)).toarray()
#Task 1
model = Sequential() 
#Convolution
model.add(Conv2D(32, kernel_size=(3, 3), strides=1, padding='same', input_shape= (32, 32, 3), activation='relu'))
#Pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
#Adding a second convolutional layer
model.add(Conv2D(64, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#Adding a third convolutional layer
model.add(Conv2D(128, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#Flattening
model.add(Flatten())
#Task 2
#Drouput
#model.add(Dropout(0.5))
#Task 3,4
#Full connection
model.add(Dense(128,activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
                #))
model.add(Dense(10, activation = 'softmax'))
model.summary()
#Compile the model

opt = SGD(lr=0.01,decay=1e-6, momentum=0.8, nesterov=True)
model.compile(
    loss='categorical_crossentropy',
    optimizer=opt, 
    metrics=['accuracy']
)
#train the mdoel
history = model.fit(train_norm,y_train,batch_size=128,epochs=10,validation_data=(val_norm,y_val),verbose=1)
#make predictions (will give a probability distribution)
pred = model.predict(test_norm)
#now pick the most likely outcome
pred = np.argmax(pred,axis=1)
y_compare = np.argmax(y_test,axis=1) 
#and calculate accuracy
score = metrics.accuracy_score(y_compare, pred)
#plot,represent loss and accuracy
show_final_history(history)
model_score =model.evaluate(test_norm,y_test)  
print("Loss:",model_score[0])
print("Accuracy:",model_score[1])
#Confusion matrix
#cm = metrics.confusion_matrix(raw_y_test, pred)
#cm_normalized = np.divide(cm.astype('float'),cm.sum(axis=1)[:, np.newaxis], where=cm.sum(axis=1)[:, np.newaxis]!=0)
#figure
#plt.figure(figsize=(9,9))
#sns.heatmap(cm_normalized, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
#plt.ylabel('Actual label');
#plt.xlabel('Predicted label');
#all_sample_title = 'Accuracy Score: {:.3f}'.format(score) 
#plt.title(all_sample_title, size = 15);