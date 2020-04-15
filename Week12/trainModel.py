#config>>>
isDense = True
epochs = 40
#config<<<
#fixed>>>
isCNN = not isDense
#fixed<<<

#print the config
if(isCNN):
    print('This is CNN model training part.')
elif(isDense):
    print('This is Dense model training part.')
print('The number of epochs are: '+str(epochs))


from keras import models
from keras import layers

#making the neural network>>>
model = models.Sequential()

if(isCNN):
    #covnet
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    #connect to the original part model
    model.add(layers.Flatten())
    #drop out
    model.add(layers.Dropout(0.5))
    # The original part
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
elif(isDense):
    #dense net
    model.add(layers.Dense(1, activation='relu', input_shape=(28,28,1)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512,activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.layers[0].trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
#making the neural network<<<

model.summary()

import numpy as np
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import os, shutil

#data loading
DatasetDir = "/home/chen/LiChiChang/2020SpringThesis/dataset/mnist/mnist" #pls indicate your path
SmallDataDir = "/home/chen/LiChiChang/2020SpringThesis/dataset/mnist/shrink" #pls indicate your path

trainDir = os.path.join(SmallDataDir, 'train')
validationDir = os.path.join(SmallDataDir, 'validation')

if os.path.exists(SmallDataDir):
    shutil.rmtree(SmallDataDir)

os.mkdir(SmallDataDir)
os.mkdir(validationDir)
os.mkdir(trainDir)

def loadData(src, dst, format, start, end):
    dataList = []
    for i in range(start, end):
        dataList.append(format.format(i))

    for data in dataList:
        for j in range(10):
            if(not os.path.exists(os.path.join(dst, str(j)))):
                os.mkdir(os.path.join(dst, str(j)))
            srcData = os.path.join(os.path.join(src, str(j)), data)
            dstData = os.path.join(os.path.join(dst, str(j)),data)
            shutil.copyfile(srcData, dstData)

loadData(DatasetDir, trainDir, '{}.jpg', 0, 5299)
loadData(DatasetDir, validationDir, '{}.jpg', 5300, 6300)

#augmentation function
DataGen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=False,)

trainGenerator = DataGen.flow_from_directory(trainDir, batch_size=100,target_size=(28,28),color_mode='grayscale',class_mode='categorical')
validationGenerator = DataGen.flow_from_directory(validationDir, batch_size=100,target_size=(28,28),color_mode='grayscale', class_mode='categorical')

history = model.fit_generator(trainGenerator, steps_per_epoch=100, epochs=epochs, validation_data=validationGenerator, validation_steps=50)

if(isCNN):
    model.save('mnistEpochs'+str(epochs)+'CNN'+'.h5')
elif(isDense):
    model.save('mnistEpochs'+str(epochs)+'DENSE'+'.h5')