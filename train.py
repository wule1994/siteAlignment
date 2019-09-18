
# coding: utf-8

# In[1]:


import numpy as np
from keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Dropout
from keras.models import Model,load_model
import scipy.io as sio  
from skimage.transform import resize
from keras import optimizers
import tensorflow as tf


# In[2]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 

#Note: we have a pretrained model in the 'model' folder. 
#You can directly run the 'demo.py' instead of training the new model.

# In[3]:


def VGG_16Model(input_shape):
    
    X_input = Input(input_shape)
    print('input_shape: ',X_input.shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1',trainable = False)(X_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2',trainable = False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    print('Block 1:',x.shape)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1',trainable = False)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2',trainable = False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    print('Block 2:',x.shape)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1',trainable = False)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2',trainable = False)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3',trainable = False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    print('Block 3:',x.shape)

    x = Conv2D(16, (7, 7), activation='relu', padding = 'valid',name = 'csw_block1_conv1')(x)
    print(x.shape)
    x = BatchNormalization(axis = 3, name = 'csw_block1_bn1')(x)
    x = Conv2D(16, (7, 7), activation='relu', padding = 'valid',name = 'csw_block1_conv2')(x)
    x = BatchNormalization(axis = 3, name = 'csw_block1_bn2')(x)
    x = Conv2D(16, (7, 7), activation='relu', padding = 'valid',name = 'csw_block1_conv3')(x)
    print(x.shape)
    x = BatchNormalization(axis = 3, name = 'csw_block1_bn3')(x)
    x = MaxPooling2D((2, 2), strides=(1, 1), name='csw_block1_pool')(x)
    print(x.shape)

    # Block 2
    x = Conv2D(16, (7, 7), activation='relu', padding = 'valid',name = 'csw_block2_conv1')(x) #44
    print(x.shape)
    x = MaxPooling2D((2, 2), strides=(1, 1), name='csw_block2_pool')(x) #2
    print(x.shape)
    
    # convert fc connection to convnet

    x = Conv2D(128,(2,2),strides = (1, 1),activation='relu',name='csw_fc1')(x)
    x = Dropout(0.8)(x)
    x = Conv2D(256,(1,1),strides = (1, 1),activation='relu',name='csw_fc2')(x) # 1*1*11
    x = Conv2D(6,(1,1),strides = (1, 1),activation='softmax',name='csw_fc3')(x) # 1*1*11
    model = Model(inputs = X_input, outputs = x)
    return model

def interpolation(data,bigNum):
    if data.shape[-1] != 3:
        return None
    
    m = len(data)
    data_interpolation = np.zeros((m,bigNum,bigNum,3))
    for i in range(m):
        data_interpolation[i] = resize(data[i], (bigNum, bigNum,3), mode='symmetric')
    return data_interpolation

# In[16]:


fileDir = './data'
path=fileDir + '/data_train.mat'
data=sio.loadmat(path)
X_train = data['data_train']


path=fileDir + '/label_train'
data=sio.loadmat(path)
label_train = ((data['label_train']).T )

path=fileDir + '/data_val'
data=sio.loadmat(path)
X_val = data['data_test']

path=fileDir + '/label_val'
data=sio.loadmat(path)
label_val = ((data['label_test']).T )


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

clsnum = 6
Y_train_temp = (convert_to_one_hot(label_train, clsnum)).T
Y_val_temp = (convert_to_one_hot(label_val, clsnum)).T
Y_train = Y_train_temp.reshape(Y_train_temp.shape[0],1,1,Y_train_temp.shape[1])
Y_val = Y_val_temp.reshape(Y_val_temp.shape[0],1,1,Y_val_temp.shape[1])


X_train_interpolation = interpolation(X_train,224)


SemgModel = VGG_16Model((224,224,3))
# load vgg16 weights
SemgModel.load_weights('./model/vgg16_weights.h5',by_name = True)

adam = optimizers.Adam(lr=0.001)
SemgModel.compile(optimizer =adam, loss ="categorical_crossentropy",metrics =["accuracy"])
SemgModel.fit(x=X_train_interpolation,y=Y_train,epochs=10,batch_size=32,shuffle=True,validation_split=0.05)

adam = optimizers.Adam(lr=0.0001)
SemgModel.compile(optimizer =adam, loss ="categorical_crossentropy",metrics =["accuracy"])
SemgModel.fit(x=X_train_interpolation,y=Y_train,epochs=15,batch_size=32,shuffle=True,validation_split=0.05)

adam = optimizers.Adam(lr=0.00001)
SemgModel.compile(optimizer =adam, loss ="categorical_crossentropy",metrics =["accuracy"])
SemgModel.fit(x=X_train_interpolation,y=Y_train,epochs=15,batch_size=32,shuffle=True,validation_split=0.05)

SemgModel.save_weights('./model/newModel.h5') #save model

X_test_interpolation = interpolation(X_val,224)
preds = SemgModel.evaluate(x =X_test_interpolation,y =Y_val)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
