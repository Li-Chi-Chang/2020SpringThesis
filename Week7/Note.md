# Week7 Cat Dog example

## Step1: Download dataset

1. remind that you need to have a kaggle account
2. download dataset via link <https://www.kaggle.com/c/3362/download-all>
3. unzip them and put this dataset to dataset folder
    * the folder tree is:
      * dataset
        * test1
          * ...
        * train
          * ...
        * sampleSubmission.csv
4. run the code in code/1.loadData.py

    ```python
    import os, shutil

    DatasetDir = "/home/chen/LiChiChang/2020SpringThesis/Week7/dataset/original/train"
    #pls indicate your path
    SmallDataDir = "/home/chen/LiChiChang/2020SpringThesis/Week7/dataset/shrink"
    #pls indicate your path

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

    print('total training cat images:', len(os.listdir(trainCatDir)))
    print('total training dog images:', len(os.listdir(trainDogDir)))
    print('total validation cat images:', len(os.listdir(validationCatDir)))
    print('total validation dog images:', len(os.listdir(validationDogDir)))
    print('total test cat images:', len(os.listdir(testCatDir)))
    print('total test dog images:', len(os.listdir(testDogDir)))
    ```

   * and the result is shown below

    ```bash
    total training cat images: 1000
    total training dog images: 1000
    total validation cat images: 500
    total validation dog images: 500
    total test cat images: 500
    total test dog images: 500
    ```

## Step2: Build a convnet

run the code at code/2.convnet.py

```python
from keras import layers
from keras import models
from keras import optimizers

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3),activation='relu', input_shape=(150,150,3)))
'''
model.add(layers.Conv2D(32, (3,3),activation='relu', input_shape=(150,150,3)))
Conv2D => data in dataset is a 2D data(pics)
32 => the output is 32 filters
(3,3) => the window size is 3X3

The input shape is an arbitary value(writen in page 134)
After this layer, the output shape is 148,148,32
'''
model.add(layers.MaxPooling2D( (2,2) ))
'''
After a conv layer, the following layer is a max pooling layer
and it will choose 1 in 2 data at each dimention(width and height)

So the output shape is 74,74,32
'''
model.add(layers.Conv2D(64, (3,3),activation='relu'))
'''
change the depth from 32 to 64
'''
model.add(layers.MaxPooling2D( (2,2) ))
'''
the output shape is 36,36,64
'''

model.add(layers.Conv2D(128, (3,3),activation='relu'))
'''
change the depth from 64 to 128
'''
model.add(layers.MaxPooling2D( (2,2) ))
'''
the output shape is 17,17,128
'''

model.add(layers.Conv2D(128, (3,3),activation='relu'))
model.add(layers.MaxPooling2D( (2,2) ))
'''
the output shape is 7,7,128
'''

model.add(layers.Flatten())
'''
input shape is 7*7*128=6272
'''

model.add(layers.Dense(512, activation='relu'))
'''
connect back to dense net
'''
model.add(layers.Dense(1, activation='sigmoid'))
'''
The last layer is only one node because it is a binary classification
because of the same reason, the activation function is 'sigmoid'
'''

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

print('step 2 is complete')
```

The result(model.summary()) cannot be shown because the size is too large, but it supposes like:

```bash
>>> model.summary()
Model: "sequential_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_9 (Conv2D)            (None, 148, 148, 32)      896
_________________________________________________________________
max_pooling2d_9 (MaxPooling2 (None, 74, 74, 32)        0
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 72, 72, 64)        18496
_________________________________________________________________
max_pooling2d_10 (MaxPooling (None, 36, 36, 64)        0
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 34, 34, 128)       73856
_________________________________________________________________
max_pooling2d_11 (MaxPooling (None, 17, 17, 128)       0
_________________________________________________________________
conv2d_12 (Conv2D)           (None, 15, 15, 128)       147584
_________________________________________________________________
max_pooling2d_12 (MaxPooling (None, 7, 7, 128)         0
_________________________________________________________________
flatten_3 (Flatten)          (None, 6272)              0
_________________________________________________________________
dense_5 (Dense)              (None, 512)               3211776
_________________________________________________________________
dense_6 (Dense)              (None, 1)                 513
=================================================================
Total params: 3,453,121
Trainable params: 3,453,121
Non-trainable params: 0
_________________________________________________________________
```

## Step3: Data preprocessing

Using data **generator** (Chinese explaination: <https://openhome.cc/Gossip/Python/YieldGenerator.html>)

```python
from keras.preprocessing.image import ImageDataGenerator

trainDataGen = ImageDataGenerator(rescale=1./255)
testDataGen = ImageDataGenerator(rescale=1./255)

trainGenerator = trainDataGen.flow_from_directory(trainDir,target_size=(150,150), batch_size=20, class_mode='binary')
validationGenerator = trainDataGen.flow_from_directory(validationDir,target_size=(150,150), batch_size=20, class_mode='binary')
'''
>>> len(trainGenerator)
100
each data in trainGenerator is (20,150,150,3)
it means that each data has 20 records, each of them is (150,150,3) like our input
>>> len(validationGenerator)
50
'''

print('step 3 is complete')
```

## Step4: Fit model and safe it as a file

Because we use generator at previous step, using fit_generator instead of fit at this part.

```python
history = model.fit_generator(trainGenerator, steps_per_epoch=100, epochs=30, validation_data=validationGenerator, validation_steps=50)
'''
Because we use generator, so we need to indicate the steps, or it will fit for infinite.
steps_per_epoch = 2000(amont of dataset)/20(each generator content)
The same reason to validation data.
validation_steps = 1000/20
'''
model.save('catDogClassifySmallDataset.h5')

print("step 4 is complete")
```

## Step5: show the summary

```python
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

print("step 5 is complete")
```

The result is shown below:
![summary.png](2020-03-03103046.png 'summary')

* Why the loss function has value higher than 1?
  * pls ref these links
    * <https://blog.csdn.net/legalhighhigh/article/details/81409551>
    * <https://zh.wikipedia.org/wiki/%E6%90%8D%E5%A4%B1%E5%87%BD%E6%95%B8>

Because our dataset is quite small, we cannot have higher accurancy.

## Step6: Do some update

### data augmentation(數據擴充)

We try to avoid overfitting especially we have small dataset. Data augmentation is a solution try to generate more train data from existing training data samples. These generated samples are from random tranformations that yield believable-looking image.

The propose of this step is trying to avoid fitting the model using same data. It can be generalization better and expose data more.

There is a example of data augmentation:

```python
'''
Example of Data Augmentation
'''

from keras.preprocessing import image

datagen = ImageDataGenerator(rotation_range=40,width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

fnames = [os.path.join(trainCatDir, fname) for fname in os.listdir(trainCatDir)]
img_path = fnames[5]#Randomly pick a picture from cats training dataset

img = image.load_img(img_path, target_size=(150, 150))
x = image.img_to_array(img)#make the input img size to fit the model and transfer img to array

x = x.reshape((1,) + x.shape)#add on feature to fit batch format

i = 0
for batch in datagen.flow(x, batch_size=1):#do the data augmentation several times(here is 4) and show the pictures
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()
```

The result is shown below:
![resultEx6.png](2020-03-03150529.png 'resultEx6')

### The model also do some update

* It is added a layer called "dropout"
This layer is try to drop some update information.

The model is shown below:

```python
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
```

the model summary:

```bash
Model: "sequential_4"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_13 (Conv2D)           (None, 148, 148, 32)      896
_________________________________________________________________
max_pooling2d_13 (MaxPooling (None, 74, 74, 32)        0
_________________________________________________________________
conv2d_14 (Conv2D)           (None, 72, 72, 64)        18496
_________________________________________________________________
max_pooling2d_14 (MaxPooling (None, 36, 36, 64)        0
_________________________________________________________________
conv2d_15 (Conv2D)           (None, 34, 34, 128)       73856
_________________________________________________________________
max_pooling2d_15 (MaxPooling (None, 17, 17, 128)       0
_________________________________________________________________
conv2d_16 (Conv2D)           (None, 15, 15, 128)       147584
_________________________________________________________________
max_pooling2d_16 (MaxPooling (None, 7, 7, 128)         0
_________________________________________________________________
flatten_4 (Flatten)          (None, 6272)              0
_________________________________________________________________
dropout_3 (Dropout)          (None, 6272)              0
_________________________________________________________________
dense_7 (Dense)              (None, 512)               3211776
_________________________________________________________________
dense_8 (Dense)              (None, 1)                 513
=================================================================
Total params: 3,453,121
Trainable params: 3,453,121
Non-trainable params: 0
_________________________________________________________________
```

### Data preprocessing(Data augmentation)

```python
trainDataGen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True,)

testDataGen = ImageDataGenerator(rescale=1./255)

trainGenerator = trainDataGen.flow_from_directory(trainDir, target_size=(150,150), batch_size=32, class_mode='binary')

validationGenerator = testDataGen.flow_from_directory(validationDir, target_size=(150,150), batch_size=32, class_mode='binary')

history = model.fit_generator(trainGenerator, steps_per_epoch=100, epochs = 100, validation_data=validationGenerator, validation_steps=50)

model.save("catDogClassfiyDataset2.h5")
```

Based on the original dataset, we create similar data and feed into this model. Now we can see how to create it.

```python
trainDataGen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True,)
```

* rescale: change every piexl value between +-1/255 times.
* rotation_range: left or right rotation between +-20 degree.
* width_shift_range / height_shift_range: shift the image horizontally or vertically the shift range is 20% of height or width
* shear_range: X or Y aisx shift and other asix fix, and the range is 20%
* zoom_range: zoom in or out with the scale 20%
* horizontal_flip: flip the picture horizontally
**Ref link<zhuanlan.zhihu.com/p/30197320>**

## Questions

1. Does high validation loss mean overfitting?
   * In other words, what does it means high validation loss?
   * Ans: **No, the loss function tells this model is good or bad.**
2. The second version of convnet has a layer called 'dropout'. I searched on the internet and found that it is a technique for fight for overfitting. What is the mechanism? It said that on the wiki: by drop out some of units to against overfitting.
    * **my understanding** is that this layer randomly chooses 50%(0.5) units and drops the update.
    * Ans: **No, while training, every epoch this layer dropout(set to 0) a half units' data.**
    * anther question about it. Does dropout occur at back propagation?

* Other things need to discuss:
  * CS232 has 2 options to take back points. But few students contact me to discuss about it.
  * About grading, can I have an announcement for the request of regrading in slack for both courses?
