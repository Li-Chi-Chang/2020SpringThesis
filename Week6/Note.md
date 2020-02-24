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
&nbsp;&nbsp;&nbsp;&nbsp;
