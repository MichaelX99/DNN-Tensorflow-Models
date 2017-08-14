https://arxiv.org/abs/1608.06993

This is the last classification network that I will write this summer before I go back to start my Masters research in detection networks.  Since I started to use TFRecord files for this I had to do a bunch of fumbling and testing before I got a working version of an efficient input pipeline.  I first tried to write my own and loosely following online blog tutorials however I could never get my volatile-gpu memory maxed as close to 100% as possible on all 3 of my gpus so I had to look elsewhere.  Finally I found the Inception tutorial code on tensorflows github and used that in conjuction with the code I had used in ResNet implementation that was largely based on the cifar10 tutorial code.

I also didn't want to type out every single layer like I have been in the past so I used loops for the first time to define the computational graph.

Follow all the steps in my ImageNet directory to set up the TFRecord files.

Run python train.py
