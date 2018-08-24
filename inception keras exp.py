import numpy as np
import pandas as pd
import os
SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
from random import randint
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.layers import Flatten, Dense, Dropout,BatchNormalization
from keras.layers import Input, concatenate
from keras.models import Model,load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model,np_utils
from keras import regularizers
import keras.metrics as metric

from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.metrics import top_k_categorical_accuracy

WEIGHT_DECAY=0.0005

#normalization
def conv2D_norm_l2(inpt,filters,kernel_size,strides=(1,1),padding='same',activation='relu',use_bias=True,
    kernel_initializer='glorot_uniform',norm=True,bias_initializer ='zero',weight_decay=WEIGHT_DECAY):
    kernel_regularizer=regularizers.l2(weight_decay)
    bias_regularizer=regularizers.l2(weight_decay)
    x = Conv2D(filters,(1,1),strides = strides ,padding=padding,activation=activation,use_bias=use_bias,
        bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer)(inpt)
    if norm:
        x=BatchNormalization()(x)
    return x

def inception_module(x,filter_set,padding='same',activation='relu',use_bias=True,
    kernel_initializer='glorot_uniform',bias_initializer='zeros'):

    size_set = [[(1,1)],[(1,1),(3,3)],[(1,1),(5,5)],[(1,1)]]
    endpoint = []
    for i in range(len(filter_set)):
        inp = x
        if i == 3 :
            # last layer
            inp = pathway4=MaxPooling2D(pool_size=(3,3),strides=1,padding=padding)(inp)
        for ii in range(len(filter_set[i])):
            inp = Conv2D(filter_set[i][ii],size_set[i][ii],strides = 1 ,padding=padding,activation=activation,
            use_bias=use_bias, kernel_initializer=kernel_initializer,bias_initializer=bias_initializer)(inp)
        endpoint.append(inp)

    return concatenate(endpoint,axis=3)

def Inception_v1():
    inp = Input(shape = (224, 224, 3), name = "input_inception")

    x=conv2D_norm_l2(inp,64,(7,7),2,padding='same',norm=False)
    x=MaxPooling2D(pool_size=(3,3),strides=2,padding='same')(x)
    x=BatchNormalization()(x)
    x=conv2D_norm_l2(x,64,(1,1),1,padding='same',norm=False)
    x=conv2D_norm_l2(x,192,(3,3),1,padding='same',norm=True)
    x=MaxPooling2D(pool_size=(3,3),strides=2,padding='same')(x)

    x=inception_module(x,filter_set=[[64],[96,128],[16,32],[32]]) #3a
    x=inception_module(x,filter_set=[[128],[128,192],[32,96],[64]]) #3b
    x=MaxPooling2D(pool_size=(3,3),strides=2,padding='same')(x)
    
    x=inception_module(x,filter_set=[[192],[96,208],[16,48],[64]]) #4a
    x=inception_module(x,filter_set=[[160],[112,224],[24,64],[64]]) #4b
    x=inception_module(x,filter_set=[[128],[128,256],[24,64],[64]]) #4c
    x=inception_module(x,filter_set=[[112],[144,288],[32,64],[64]]) #4d
    x=inception_module(x,filter_set=[[256],[160,320],[32,128],[128]]) #4e
    x=MaxPooling2D(pool_size=(3,3),strides=2,padding='same')(x)

    x=inception_module(x,filter_set=[[256],[160,320],[32,128],[128]]) #5a
    x=inception_module(x,filter_set=[[384],[192,384],[48,128],[128]]) #5b
    x=AveragePooling2D(pool_size=(7,7),strides=1,padding='valid')(x)

    DROPOUT=0.5
    x=Flatten()(x)
    x=Dropout(DROPOUT)(x)
    x=Dense(2,activation='softmax')(x)
    model=Model(input=inp,output=x)
    return model

def acc_top2(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

if __name__=='__main__':
    batch_size = 2
    height, width = 224, 224
    inputs = np.random.rand(batch_size, height, width, 3)
    Y = np.array([[1,0s],[0,1]], np.uint8)
    model = Inception_v1()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[acc_top2])
    model.summary()
    # Save a PNG of the Model Build
    plot_model(model, to_file=SCRIPT_PATH+'/Inception_v1.png')
    print('Model Compiled') 
    earlystopper = EarlyStopping(patience=5, verbose=1)
    checkpointer = ModelCheckpoint('model-tgs-tf-1.h5', verbose=1, save_best_only=True)
    results = model.fit(inputs, Y, batch_size=1, epochs=50, 
                        callbacks=[earlystopper, checkpointer])
