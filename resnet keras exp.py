import numpy as np
import pandas as pd
import os
SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
from random import randint
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, \
    Activation, ZeroPadding2D
from keras.layers import add, Flatten
from keras.utils import plot_model
from keras.metrics import top_k_categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.callbacks import ModelCheckpoint,EarlyStopping

import os

def bottleneck_Block(inpt,filters,strides=(1,1),deepnotsame=False):
    x = Conv2D(filters[0],(1,1),strides = strides ,padding='same',activation='relu')(inpt)
    x = BatchNormalization(axis=3)(x)
    x = Conv2D(filters[1],(3,3),padding='same',activation='relu')(x)
    x = BatchNormalization(axis=3)(x)
    x = Conv2D(filters[2],(1,1),padding='same',activation='relu')(x)
    x = BatchNormalization(axis=3)(x)
    if deepnotsame:
        shortcut = Conv2D(filters[2], (1,1), strides = strides ,padding='same',activation='relu')(inpt)
        shortcut = BatchNormalization(axis=3)(shortcut)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x

def resnet_v2_50():
    inp = Input(shape = (224, 224, 3), name = "input_v2_50")
    x = BatchNormalization(axis=3,name='in_v2_n1')(inp)
    x = ZeroPadding2D((3,3))(x)
    x = Conv2D(64,(7,7), strides = (2,2),activation='relu',padding='valid')(x)
    x = MaxPooling2D((3, 3),strides = (2,2),padding='same')(x)
    rn_set = {}
    rn_set['filterset'] = [[64,64,256],[128, 128, 512],[256, 256, 1024],[512, 512, 2048]]
    rn_set['strideset'] = [(1,1),(2,2),(2,2),(2, 2)]
    rn_set['layers'] = [3,4,6,3]
    for i in range(4):
        filter_n, stride_n, layers_n = rn_set['filterset'][i], rn_set['strideset'][i], rn_set['layers'][i]
        for ii in range(layers_n): 
            if ii == 0: 
                x = bottleneck_Block(x,filter_n,strides=stride_n,deepnotsame=True)
            else:
                x = bottleneck_Block(x,filter_n)
    x = AveragePooling2D(pool_size = (7,7))(x)
    x = Flatten()(x)
    x = Dense(2,activation='softmax')(x)
    model = Model(inputs=inp,outputs=x)
    return model

def acc_top2(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

if __name__=='__main__':
    batch_size = 2
    height, width = 224, 224
    inputs = np.random.rand(batch_size, height, width, 3)
    Y = np.array([[0,1],[0,1]], np.uint8)
    # print(Y.shape)
    model = resnet_v2_50()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[acc_top2])
    model.summary()
    plot_model(model, to_file=SCRIPT_PATH+'/resnet.png')
    print('Model Compiled') 
    earlystopper = EarlyStopping(patience=5, verbose=1)
    checkpointer = ModelCheckpoint('model-tgs-tf-1.h5', verbose=1, save_best_only=True)
    results = model.fit(inputs, Y, batch_size=1, epochs=50, 
                        callbacks=[earlystopper, checkpointer])


