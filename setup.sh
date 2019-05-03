#!/bin/bash

mkdir data
wget https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/flower.npy
mv flower.npy data
wget http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
tar -xvzf 102flowers.tgz
mv jpg data/jpg

