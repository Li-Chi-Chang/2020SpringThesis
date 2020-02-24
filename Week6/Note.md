# Deep learning Note

Name: Li-Chi Chang
Professor: Dr. Chen
Date: Feb/19-25
Coverage: Chap 4.4

## ToDo list

1. 2nd video of MIT camp
2. lab on google cloud of MIT camp
3. chap 4.4 - 4.5 & chap 5
4. do some examples

## DL Bible

### Ch 4.4 overfitting and underfitting

&nbsp;&nbsp;&nbsp;&nbsp;The processing of fighting overfitting this way is called regularization. We can look to chap 3.4 and find examples. **The simplest way to prevent overfitting is to reduce the size of the model.** the number of learnable parameters in the model (which is determined by the number of layers and the number of units per layer) The general workflow to find an appropriate model size is **to start with relatively few layers and parameters, and increase the size of the layers or add new layers until you see diminishing returns with regard to validation loss.**

### Ch 4.5 The universal workflow of machine learning

1. problem definition
2. evaluation
3. feature engineering
4. fighting overfitting

#### Defining the problem and assembling a dataset

1. What's the input data format?
2. What's the prediction you try to find?
3. What type of problem are you facing?
   1. Binary classification?
   2. Multiclass classification?
   3. Scalar regression?
   4. Vector regression?
   5. Multiclass, multilabel classification?
   6. clustering?
   7. generation?
   8. reinforcement learning?

You have input and assume that:

1. Your outputs can be predicted given your inputs.
2. Your available data is sufficiently informative to learn the relationship between inputs and outputs.

#### Choosing a measure of success

To achieve success, you must define what you mean by success—accuracy? Precision and recall? Customer-retention rate? Your metric for success will guide the choice of a loss function: what your model will optimize.

#### Deciding on an evaluation protocol

* hold-out validation set
* K-fold cross-validation
* iterated K-fold validation

#### Preparing your data

* Data should be formatted as tensors.
* The values taken by these tensors should usually be scaled to small values: for example, in the [-1, 1] range or [0, 1] range.
* Data should be normalized.
* do some feature engineering, especially for small-data problems.

#### Developing a model that does better than a baseline

you need to make three key choices to build your first working model:

* Last-layer activation—This establishes useful constraints on the network’s output.
* Loss function—This should match the type of problem you’re trying to solve.
* Optimization configuration—What optimizer will you use? What will its learning rate be? In most cases, it’s safe to go with rmsprop and its default learning rate.
![Table 4.1](4.1.png "Table 4.1")

#### Scaling up: developing a model that overfits

To figure out how big a model you’ll need, you must develop a model that overfits.
This is fairly easy:

1. Add layers.
2. Make the layers bigger.
3. Train for more epochs.

#### Regularizing your model and tuning your hyperparameters

This step will take the most time. Modifing your model.

1. Add dropout.
2. Try different architectures: add or remove layers.
3. Add L1 and/or L2 regularization.
   **I don't know what is it.**
   **(<https://en.wikipedia.org/wiki/Regularization_(mathematics)>)**
4. Try different hyperparameters (such as the number of units per layer or the learning rate of the optimizer) to find the optimal configuration.
5. Optionally, iterate on feature engineering: add new features, or remove features that don’t seem to be informative.

        Need to notice that. Information leak will occur while multiple times iteration.

If you find a satisfactory model, you can start to train it. If the result of testing set is not as good as the validation set, it means that it is overfitting. Once you meet this issue, you can consider **iterated K fold validation**.

## MINST Example

### The function is below

```python
def mnist2LayersTest(Units,Epochs,Layers):
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

   #making the neural network>>>
   network = models.Sequential()
   network.add(layers.Dense(Units, activation='relu', input_shape=(28 * 28,)))
   for i in range(Layers):
      network.add(layers.Dense(Units, activation='relu'))
   network.add(layers.Dense(10, activation='softmax'))

   network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
   #making the neural network<<<

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
   network.fit(train_images, train_labels, epochs=Epochs, batch_size=128)
   #training<<<

   #testing>>>
   test_loss, test_acc = network.evaluate(test_images, test_labels)
   #testing<<<

   return test_acc
```

### This is the experiment of the number of Epochs

```python
#epochs find the peak
import matplotlib.pyplot as plt

acclist=[]
for i in range(11):
    acclist.append(mnist2LayersTest(2048,i,0))

plt.plot(acclist)
plt.show()

print(acclist)
```

#### This is the Epochs' result in the array

      In the loops
         0.053700000047683716
         0.9686999917030334
         0.9750999808311462
         0.979200005531311
         0.977400004863739
         0.98089998960495
         0.9815999865531921
         0.9818999767303467
         0.9810000061988831
         0.9818999767303467
         0.9807999730110168

#### Epoch part Summary

In this experiment we can see that, more epochs make more accurancy.

### This is the experiment of the number of Units

```python
#Units find the peak
import matplotlib.pyplot as plt

acclist=[]
for i in range(15):
    acclist.append(mnist2LayersTest(2**i,3,0))

plt.plot(acclist[1:])
plt.show()

print(acclist)
```

#### This is the Units' result in the array

      In the loops
         0.2240999937057495
         0.5679000020027161
         0.8130000233650208
         0.90420001745224
         0.9319000244140625
         0.9488000273704529
         0.9581000208854675
         0.9664000272750854
         0.9660000205039978
         0.9764000177383423
         0.9771999716758728
         0.977400004863739
         0.9815000295639038
         0.980400025844574
         0.9793000221252441

#### Unit part Summary

In this experiment we can see that,
