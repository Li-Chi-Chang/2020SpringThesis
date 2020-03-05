import os, shutil
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

DatasetDir = "/home/chen/LiChiChang/2020SpringThesis/dataset/CatDog/original/train" #pls indicate your path
SmallDataDir = "/home/chen/LiChiChang/2020SpringThesis/dataset/CatDog/shrink" #pls indicate your path

trainDir = os.path.join(SmallDataDir, 'train')
trainCatDir = os.path.join(trainDir, 'cat')
trainDogDir = os.path.join(trainDir, 'dog')

validationDir = os.path.join(SmallDataDir, 'validation')
validationCatDir = os.path.join(validationDir, 'cat')
validationDogDir = os.path.join(validationDir, 'dog')

testDir = os.path.join(SmallDataDir, 'test')
testCatDir = os.path.join(testDir, 'cat')
testDogDir = os.path.join(testDir, 'dog')

if os.path.exists(SmallDataDir):
    shutil.rmtree(SmallDataDir)

os.mkdir(SmallDataDir)

os.mkdir(trainDir)
os.mkdir(trainCatDir)
os.mkdir(trainDogDir)

os.mkdir(validationDir)
os.mkdir(validationCatDir)
os.mkdir(validationDogDir)

os.mkdir(testDir)
os.mkdir(testCatDir)
os.mkdir(testDogDir)

def loadData(src, dst, format, start, end):
    datalist = [format.format(i) for i in range(start,end)]
    for data in datalist:
        srcData = os.path.join(src, data)
        dstData = os.path.join(dst, data)
        shutil.copyfile(srcData, dstData)

loadData(DatasetDir, trainCatDir, 'cat.{}.jpg', 0, 1000)
loadData(DatasetDir, validationCatDir, 'cat.{}.jpg', 1000, 1500)
loadData(DatasetDir, testCatDir, 'cat.{}.jpg', 1500, 2000)

loadData(DatasetDir, trainDogDir, 'dog.{}.jpg', 0, 1000)
loadData(DatasetDir, validationDogDir, 'dog.{}.jpg', 1000, 1500)
loadData(DatasetDir, testDogDir, 'dog.{}.jpg', 1500, 2000)

print('step 1 is complete(make small dataset)')

'''================================================================================'''

model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))
model.add(layers.MaxPooling2D( (2,2) ))

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D( (2,2) ))

model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D( (2,2) ))

model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D( (2,2) ))

model.add(layers.Flatten())
#Only add this layer comparing to the previous model
model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

print("step 7 is complete(make a dropout model)")

'''================================================================================'''

trainDataGen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True,)

testDataGen = ImageDataGenerator(rescale=1./255)

trainGenerator = trainDataGen.flow_from_directory(trainDir, target_size=(150,150), batch_size=32, class_mode='binary')

validationGenerator = testDataGen.flow_from_directory(validationDir, target_size=(150,150), batch_size=32, class_mode='binary')

history = model.fit_generator(trainGenerator, steps_per_epoch=100, epochs = 100, validation_data=validationGenerator, validation_steps=50)

model.save("catDogClassfiyDataset2.h5")

print("step 8 is complete(train and save the model)")

'''================================================================================'''

import matplotlib.pyplot as plt

acc = history.history['acc']
validationAcc = history.history['val_acc']

loss = history.history['loss']
validationLoss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='training acc')
plt.plot(epochs, validationAcc, 'b', label='validation acc')
plt.title('training and validation accuracy')
plt.legend()
plt.show()

plt.plot(epochs, loss, 'bo', label='training loss')
plt.plot(epochs, validationLoss, 'b', label='validation loss')
plt.title('training and validation loss')
plt.legend()
plt.show()

print("step 5 is complete(show loss Summary)")