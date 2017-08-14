https://arxiv.org/abs/1512.03385

This was my implementation of ResNet.  This is my last code that used the feed_dict method because I did not know about different ways of designing an input pipeline until this code.  I included both the 34 and 50 layer implementations of the paper since this included the original architecture and the bottleneck architecture and any deeper architecture just included stacking more bottleneck convolutions which is trivial and not useful for my purposes.  This was before I understood what TFRecord files were so you should not use the total ImageNet directory.

1- Download ImageNet

2- Run convert.sh

3- run python train.py
