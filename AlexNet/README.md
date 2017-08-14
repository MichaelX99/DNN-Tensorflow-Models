https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks

Download and set up CIFAR10 and if you want CIFAR100, to do this I used the DIGITS download scripts but I'm sure there are a million better ways to do this.

So this isn't quite AlexNet because I trained it on CIFAR10 and not ImageNet but it was the same general idea.  The CIFAR10_AlexNet script contains much of the same code in the CIFAR10 tutorial.  The AlexNet script contains the original architecture of AlexNet not accounting for its multigpu split since I only had a single 960 at the time.  Also I had to make a few changes to the architecture because the image size of cifar and imagenet made me change the first layer since the feature map size would be too small for the fully connected layers.
