import scipy.misc as misc
from keras.datasets import mnist
from PIL import Image
import os

count = [0,0,0,0,0,0,0,0,0,0]# 0,1,2,3,4,5,6,7,8,9

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
for a in range(len(train_images)):
    #im = Image.fromarray(train_images[a])
    #if(not os.path.exists("../dataset/mnist/"+str(train_labels[a]))):
    #    os.mkdir("../dataset/mnist/"+str(train_labels[a]))
    #im.save("../dataset/mnist/"+str(train_labels[a])+"/"+str(count[train_labels[a]])+ ".jpg")
    count[train_labels[a]] = count[train_labels[a]] + 1

for a in range(len(test_images)):
    #im = Image.fromarray(test_images[a])
    #if(not os.path.exists("../dataset/mnist/"+str(test_labels[a]))):
    #    os.mkdir("../dataset/mnist/"+str(test_labels[a]))
    #im.save("../dataset/mnist/"+str(test_labels[a])+"/"+str(count[test_labels[a]])+ ".jpg")
    count[test_labels[a]] = count[test_labels[a]] + 1

print(count)