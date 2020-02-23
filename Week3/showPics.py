"""
this is a example in Chap2
it needs to be executed in local or it cannot show correctly
"""

import numpy as np
import matplotlib.pyplot as plt

f = np.load("/home/chen/.keras/datasets/mnist.npz")
train_images, train_labels = f['x_train'], f['y_train']
test_images, test_labels = f['x_test'], f['y_test']
f.close()

digit = train_images[4]
plt.imshow(digit, cmap=plt.cm.get_cmap('binary'))
plt.show()