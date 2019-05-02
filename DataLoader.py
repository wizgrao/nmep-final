import scipy
from glob import glob
import numpy as np
import urllib.request
from skimage.transform import resize
from numpy.random import shuffle
from scipy import misc
import random


def download_quick(self):
    base = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
    path = base + self.name + '.npy'
    print(path)
    urllib.request.urlretrieve(path, 'data/' + self.name + '.npy')


class ImageDataLoader:
    def __init__(self, batch_size, res=(128, 128)):
        self.res = res
        self.idx = 0
        self.batch_size = batch_size
        self.filenames = glob('./data/jpg/*')
        self.N = len(self.filenames)
        self.new_epoch()

    def new_epoch(self):
        random.shuffle(self.filenames)
        self.idx = 0

    def get_batch(self):
        imgs = np.array([self.resize_img(misc.imread(x)) for x in self.filenames[self.idx:self.idx + self.batch_size]])
        self.idx += self.batch_size
        if self.idx + self.batch_size   > self.N:
            self.new_epoch()
        return imgs

    def resize_img(self, image):
        h, w, _ = image.shape
        if h > w:
            image = image[(h-w)//2:-(h-w)//2, :]
        if w > h:
            image = image[:, (w-h)//2:-(w-h)//2]
        return resize(image, self.res)


class DrawingDataLoader:
    def __init__(self, name, batch_size, res=(128, 128)):
        self.name = name
        self.res = res
        self.idx = 0
        self.batch_size = batch_size
        self.raw = np.load("data/" + name + ".npy")
        self.N, _ = self.raw.shape

    def new_epoch(self):
        shuffle(self.raw)
        self.idx = 0

    def get_batch(self):
        imgs = np.array([self.resize_img(x) for x in self.raw[self.idx:self.idx + self.batch_size]])
        self.idx += self.batch_size
        if self.idx + self.batch_size > self.N:
            self.new_epoch()
        return imgs

    def resize_img(self, image):
        image = np.reshape(image, (28, 28))
        image = np.stack((image,)*3, axis=-1)
        return resize(image, self.res)
