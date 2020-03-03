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