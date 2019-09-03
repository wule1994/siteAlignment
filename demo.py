
# coding: utf-8


import numpy as np
from keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Dropout,Activation
from keras.models import Model,load_model
import scipy.io as sio
from skimage.transform import resize
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 


def interpolation(data,bigNum):
    # only for dim is 3
    if data.shape[-1] != 3:
        return None
    
    m = len(data)
    data_interpolation = np.zeros((m,bigNum,bigNum,3))
    for i in range(m):
        data_interpolation[i] = resize(data[i], (bigNum, bigNum,3), mode='symmetric')
    return data_interpolation

# function for image match (MAD)
def showpiclocation(img,smallimg,resultNum):
    w=img.shape[1]   #img row
    h=img.shape[0]  #img column
    n=img.shape[2] #3st dim
    fw=smallimg.shape[1]
    fh=smallimg.shape[0]
    comp_tz = np.zeros((h-fh+1,w-fw+1))
    compTemp = np.zeros((h-fh+1,w-fw+1,n)) 
    for now_n in range(n):
        for now_h in range(0,h-fh+1):
            for now_w in range(0,w-fw+1):
                compTemp[now_h,now_w,now_n]=np.sum(np.square(img[now_h:now_h+fh,now_w:now_w+fw,now_n]-smallimg[:,:,now_n]))  
    comp_tz = np.sum(compTemp,axis=2)
    sortValue = np.sort(comp_tz.flatten())
    sortIndex = np.where(comp_tz<=sortValue[resultNum-1])
    value = np.mean(sortValue[0:resultNum])
    return np.array(sortIndex),value


def interpolation_template(data,bigNum):
    if data.shape[-1] != 3:
        return None
    data_interpolation = resize(data, (bigNum, bigNum,3), mode='symmetric')
    return data_interpolation


def importTemplate(path):
    data=sio.loadmat(path)
    template8 = np.zeros((6,8,8,3))
    template8[0] = data['A2']
    template8[1] = data['A3']
    template8[2] = data['A5']
    template8[3] = data['A23']
    template8[4] = data['A345']
    template8[5] = data['WE']
    return template8


def getSpecialDim(img,dims):
    featureNum = len(dims)
    result = np.zeros((img.shape[0],img.shape[1],featureNum))
    for i in range(featureNum):
        result[:,:,i] = img[:,:,dims[i]]
    return result

def normPic(img):
    featureNum = img.shape[-1]
    result = np.zeros(img.shape)
    for n in range(featureNum):
        result[:,:,n] = (img[:,:,n]-np.min(img[:,:,n]))/np.max(img[:,:,n])
    return result


def findTopNumValue(matrix,num):
    noiseMatrix = matrix
    sortValue = np.sort(noiseMatrix.flatten())
    result = (noiseMatrix>=sortValue[-num])*1*matrix
    result = result/np.max(result)
    return result

def findclsCaliTop(matrix,num):
    normClsCalibrationMatrix = np.zeros((matrix.shape))
    for id in range(normClsCalibrationMatrix.shape[0]):
        if np.max(matrix[id])!=0 :
            normClsCalibrationMatrix[id] = matrix[id]/np.max(matrix[id])
    sumNormClsCalibrationMatrix = np.sum(normClsCalibrationMatrix,axis=0)
    result = findTopNumValue(sumNormClsCalibrationMatrix,num)
    return result


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

smallSize = 92
bigSize = 115

template8Path = './data/template8'
data=sio.loadmat(template8Path)
template8 = np.zeros((6,8,8,3))
A2 = data['A2']
template8[0] = A2
A3 = data['A3']
template8[1] = A3
A5 = data['A5']
template8[2] = A5
A23 = data['A23']
template8[3] = A23
A345 = data['A345']
template8[4] = A345
WE = data['WE']
template8[5] = WE

template224 = np.zeros((6,smallSize,smallSize,3))
for i in range(6):
    template224[i] = interpolation_template(template8[i],smallSize)
    template224[i] = normPic(template224[i])

# load model
#modelWeightName =  './model/newModel.h5'
modelWeightName =  './model/pretrained.h5' #you can use our pretrained model
SemgModel = VGG_16Model((280,280,3))
SemgModel.load_weights(modelWeightName)
 
path= './data/data_test' 
data=sio.loadmat(path)
X_test = data['data_test']
path='./data/label_test'
data=sio.loadmat(path)
label_test = data['label_test']

# step2: interpolation
X_test_Calibrate = interpolation(X_test,bigSize)
X_test_interpolation = interpolation(X_test,280)

X_test = X_test_interpolation
X_cali = X_test_Calibrate
label_test_shuffle = label_test

result = SemgModel.predict(X_test)
result_temp = np.zeros(result.shape)
for i in range(result.shape[0]):
    result_temp[i] = result[i]

predclsNN = np.argmax(result, axis=3)
temp=np.zeros((8,8))
for i in range(8):
    for j in range(8):
        subpred = predclsNN[:,i,j]
        subpred = subpred.reshape(len(predclsNN),1)
        sumNum = np.sum((subpred==label_test)*1)
        temp[i,j]=sumNum/len(predclsNN)

caliNum = 50
updateCaliNum = 10 

# step1 image match
locationMatrix = np.zeros((bigSize - smallSize + 1,bigSize - smallSize + 1))
cls = len(template224) #class
Data_calibration = X_cali[0:caliNum]

predcls = np.zeros(label_test_shuffle.shape, dtype=np.int8)   
accMatrix = np.zeros([cls,cls])
predclsNN = np.argmax(result[0:caliNum] , axis=3)
maxResult =  np.max(result_temp,axis = -1)

clsCalibrationMatrix = np.zeros((6,8,8))
calibrationMatrix = np.zeros((8,8))
onlinAcc = []
initializationNum = 1 
for i in range(caliNum):
    tempPred = predclsNN[i] 
    dims = [0]
    img = getSpecialDim(Data_calibration[i],dims)
    img = normPic(img)
    temp = np.zeros((cls,1))
    loc = np.zeros((cls,2,initializationNum))
    for clsid in range(cls): 
        locResult = showpiclocation(img,getSpecialDim(template224[clsid],dims),initializationNum)
        temp[clsid]  = locResult[1]
        loc[clsid] = locResult[0]

    clsind = np.argmin(temp)
    clsCNN = tempPred[int(loc[clsind,0,0]/3),int(loc[clsind,1,0]/3)]
    predcls[i] = clsCNN
    accMatrix[int(label_test_shuffle[i]), clsCNN]+= 1

    clsCalibrationMatrix[clsCNN] += (tempPred==clsCNN)*1*maxResult[i]


calibrationMatrix = findclsCaliTop(clsCalibrationMatrix,updateCaliNum)
# stage one over

updateCaliNum = 5
lengthY, lengthRow, lengthColumn ,cls= result.shape
#     stage two begin
for i in range(caliNum,lengthY):
    predTemp = result[i]
    tempCls = np.argmax(predTemp , axis=-1)
    maxValue = np.max(predTemp,axis=-1)

    for clsid in range(cls):
        predTemp[:,:,clsid] = np.multiply(predTemp[:,:,clsid],calibrationMatrix)
    t = np.sum(predTemp.reshape(lengthRow*lengthColumn,cls),axis=0)
    predcls[i] = np.argmax(t)
    accMatrix[int(label_test_shuffle[i]), predcls[i]]+= 1

    clsCalibrationMatrix[predcls[i]] += (tempCls==predcls[i])*maxResult[i]
    if i != caliNum and i%50==0:

        calibrationMatrix = findclsCaliTop(clsCalibrationMatrix,updateCaliNum)   
#for acc
sa = np.sum(accMatrix,axis = 1,keepdims=True)
sar = np.repeat(sa,cls,axis=1)
accPlot = np.round(accMatrix/sar,2)
correctRes = 0
for i in range(cls):
    correctRes += accMatrix[i,i]

overallAcc = correctRes/(np.sum(accMatrix))
print('overallAcc:\n',overallAcc)  
print('confusion matrix:\n',accPlot)
print('alignment matrix:\n',np.round(calibrationMatrix,2))