import scipy.misc as misc
from keras.datasets import mnist
from PIL import Image
import os
import numpy as np

def gray_to_gray(imgArr):
    digit = np.zeros((28,28))
    for i in range(28):
        for j in range(28):
            ch1 = imgArr[i][j]
            if(ch1 < 150):
                digit[i][j] = 0
            else:
                digit[i][j] = ch1
    return digit

count = [0,0,0,0,0,0,0,0,0,0]# 0,1,2,3,4,5,6,7,8,9

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
for a in range(len(train_images)):
    digit = gray_to_gray(train_images[a])
    im = Image.fromarray(digit)
    im = im.convert('RGB')
    if(not os.path.exists("../dataset/mnist/mnist/"+str(train_labels[a]))):
        os.mkdir("../dataset/mnist/mnist/"+str(train_labels[a]))
    im.save("../dataset/mnist/mnist/"+str(train_labels[a])+"/"+str(count[train_labels[a]])+ ".jpg")
    count[train_labels[a]] = count[train_labels[a]] + 1

for a in range(len(test_images)):
    digit = gray_to_gray(test_images[a])
    im = Image.fromarray(digit)
    im = im.convert('RGB')
    if(not os.path.exists("../dataset/mnist/mnist/"+str(test_labels[a]))):
        os.mkdir("../dataset/mnist/mnist/"+str(test_labels[a]))
    im.save("../dataset/mnist/mnist/"+str(test_labels[a])+"/"+str(count[test_labels[a]])+ ".jpg")
    count[test_labels[a]] = count[test_labels[a]] + 1

print(count)