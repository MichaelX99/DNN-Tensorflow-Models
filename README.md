# DNN-Tensorflow-Models

This is my git repo of the Tensorflow code that I wrote in order to prepare me for my Master's research.  I implemented as closely as possible each of the network architectures and methods used to train them that I could.  Each paper has a new feature in Tensorflow and serves as a roadmap in order to go from a Tensorflow noob to a competent user.

1- LeNet
This implements the super simple LeNet network and implements basic Python/Tensorflow mechanics.

2- AlexNet
This implements AlexNet that started the hype and moves into more complex Tensorflow mechanics such as image processing, TensorBoard usage, and a learning rate schedule.  Obviously this is a slightly larger network so I also added using a dictionary to hold the weights.

3- GoogLeNet
This implements GoogLeNet and the inception module.  This also marked my official move to ImageNet away from CIFAR10 since I finally built my new computer.  Since there are many weights in this network, I defined a class that would hold all my weights and would also initialize them with a built in function.  I also wrote functions for each of the network layer types to eliminate boilerplate code.

4- ResNet
This is a massive shift from the 3 previous networks.  This was the first network that I wrote the whole code on my new computer and therefore includes a naive implementation of a multi-gpu system.  This includes alot of the CIFAR10 tutorial code however tailored to my ability and understanding at the time.  The largest failures with this implementation is my inability to use Queue Runners which therefore makes the training time basically intractable.  However I did use some of the functions in the image portion of Tensorflow which will be the basis of the next network.

5- DenseNet
This is the last model implementation that will be of classification networks alone.  The last major change that I have implemented is the input pipeline is be a queue import of TFRecord files.  The reason for this change is because using the feed_dict method starved my gpu's of data and my network went nowhere fast.  The input pipeline is mostly copied from the Inception tutorial code since the input pipeline of the CIFAR10 tutorial code did not solve the gpu data starving problem.
