# Week7 Cat Dog example

## Step1: Download and preprocess dataset

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
```

The result(model.summary()) cannot be shown because the size is too large, but it supposes like:

```bash
>>> model.summary()
Layer           (type)          Output Shape            Param #
================================================================
conv2d_1        (Conv2D)        (None, 148, 148, 32)    896
________________________________________________________________
maxpooling2d_1  (MaxPooling2D)  (None, 74, 74, 32)      0
________________________________________________________________
conv2d_2        (Conv2D)        (None, 72, 72, 64)      18496
________________________________________________________________
maxpooling2d_2  (MaxPooling2D)  (None, 36, 36, 64)      0
________________________________________________________________
conv2d_3        (Conv2D)        (None, 34, 34, 128)     73856
________________________________________________________________
maxpooling2d_3  (MaxPooling2D)  (None, 17, 17, 128)     0
________________________________________________________________
conv2d_4        (Conv2D)        (None, 15, 15, 128)     147584
________________________________________________________________
maxpooling2d_4  (MaxPooling2D)  (None, 7, 7, 128)       0
________________________________________________________________
flatten_1       (Flatten)       (None, 6272)            0
________________________________________________________________
dense_1         (Dense)         (None, 512)             3211776
________________________________________________________________
dense_2         (Dense)         (None, 1)               513
================================================================
Total params: 3,453,121
Trainable params: 3,453,121
Non-trainable params: 0
```
