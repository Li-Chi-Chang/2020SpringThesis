# Week 8 note

## TODO list

1. using pretrain model
2. trying to predict a new picture from internet
3. the classic paper reading

## Chap 5.3 Using a pretrained convnet

This may be your first encounter with one of these cutesy model names—VGG, ResNet, Inception, Inception-ResNet, Xception, and so on; you’ll get used to them, because they will come up frequently if you keep doing deep learning for computer vision.

There are two ways to use a pretrained network: feature extraction and fine-tuning

1Q: IOT device voice detected: Sniff and spoof the voice detect using Deep learning after recording
How to defend it?

* GAN

## Ch6 Text model

### What are the domin issues

1. text is sequential. So the model needs to be designed with time.
2. variably input. the input size is not fixed.

Solution:

* RNN -> Solve the time stemp issue
* One hot encoding -> Solve the variably input

## Addversarial Essay reimplement

I found the reimplement of this essay, and I try to directly use codes.
And it is sucessful.

### found in the codes

1. bacially it is trying to use the parameters in a model
2. use the loss of a specific picture
3. find the loss value of each pixel and add them into the picture
4. make the adversarial graph

The concept is
