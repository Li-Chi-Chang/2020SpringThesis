from keras.utils import to_categorical
from keras.datasets import mnist
from keras import models
from keras import layers
import numpy as np

#(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
f = np.load("/home/chen/.keras/datasets/mnist.npz")
train_images, train_labels = f['x_train'], f['y_train']
test_images, test_labels = f['x_test'], f['y_test']
f.close()

"""
In the def of load_data() in mnist.py
I found that 
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close() 

For now the path is "/home/chen/.keras/datasets/mnist.npz"
"""

#making the neural network>>>
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#making the neural network<<<
network.summary()

#resize the inputs>>>
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
#add labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
#resize the inputs<<<

#training>>>
network.fit(train_images, train_labels, epochs=5, batch_size=128)
#training<<<

#testing>>>
test_loss, test_acc = network.evaluate(test_images, test_labels)
#testing<<<

print('test_acc:', test_acc)