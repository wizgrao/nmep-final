import os
import glob
import numpy as np
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

from tensorflow import keras

import tensorflow as tf
import matplotlib.pyplot as plt

import urllib.request
import datetime

from DataLoader import ImageDataLoader, DrawingDataLoader

from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras import backend as K
from tensorflow.keras.backend import clear_session
import gc


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class InstanceNormalization(Layer):
    """Instance normalization layer.
    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each
            instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a Sequential model.
    # Output shape
        Same shape as input.
    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](
        https://arxiv.org/abs/1607.08022)
    """
    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CycleGan:
    def __init__(self, shape, load=False, prefix=""):
        # 3 entry tuple
        self.shape = shape

        # Calculate output shape of patchgan discriminator
        patch = int(self.shape[0] / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

        # Beginning number of filters
        self.g_filters = 32
        self.d_filters = 64

        # Loss weights
        self.lambda_cycle = 10.0
        self.lambda_id = 0.1 * self.lambda_cycle

        optimizer = Adam(0.0002, 0.5)

        # Create and compile discriminators
        self.d_X = self.discriminator()
        self.d_Y = self.discriminator()
        self.g_XY = self.generator()
        self.g_YX = self.generator()

        if load:
            self.g_XY = load_model(prefix + 'gy.h5', custom_objects={'InstanceNormalization': InstanceNormalization})
            self.g_YX = load_model(prefix + 'gx.h5', custom_objects={'InstanceNormalization': InstanceNormalization})
            self.d_X = load_model(prefix + 'dx.h5', custom_objects={'InstanceNormalization': InstanceNormalization})
            self.d_Y = load_model(prefix + 'dy.h5', custom_objects={'InstanceNormalization': InstanceNormalization})
        self.d_X.compile(loss='mse',
                         optimizer=optimizer,
                         metrics=['accuracy'])
        self.d_Y.compile(loss='mse',
                         optimizer=optimizer,
                         metrics=['accuracy'])



        img_x = Input(shape=self.shape)
        img_y = Input(shape=self.shape)

        fake_y = self.g_XY(img_x, training=True)
        fake_x = self.g_YX(img_y, training=True)

        cycle_x = self.g_YX(fake_y, training=True)
        cycle_y = self.g_XY(fake_x, training=True)

        x_id = self.g_YX(img_x, training=True)
        y_id = self.g_XY(img_y, training=True)

        self.d_X.trainable = False
        self.d_Y.trainable = False

        v_x = self.d_X(fake_x, training=True)
        v_y = self.d_Y(fake_y, training=True)

        self.combined = Model(inputs=[img_x, img_y],
                              outputs=[v_x, v_y,
                                       cycle_x, cycle_y,
                                       x_id, y_id])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                              loss_weights=[1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id],
                              optimizer=optimizer)

    def generator(self):
        def conv(inputs, filters, size=4):
            x = Conv2D(filters, kernel_size=size, strides=2, padding='same')(inputs)
            x = LeakyReLU(alpha=0.2)(x)
            x = InstanceNormalization()(x)
            return x

        def deConv(inputs, skip, filters, size=4):
            x = UpSampling2D(size=2)(inputs)
            x = Conv2D(filters, kernel_size=size, strides=1, padding='same', activation='relu')(x)
            x = (InstanceNormalization())(x)
            x = Concatenate()([x, skip])
            return x

        inputs = Input(shape=self.shape)

        x1 = conv(inputs, self.g_filters)
        x2 = conv(x1, self.g_filters * 2)
        x3 = conv(x2, self.g_filters * 4)
        x4 = conv(x3, self.g_filters * 8)

        y1 = deConv(x4, x3, self.g_filters*4)
        y2 = deConv(y1, x2, self.g_filters*2)
        y3 = deConv(y2, x1, self.g_filters)
        y4 = UpSampling2D(size=2)(y3)
        outIm = Conv2D(3, kernel_size=4, strides=1, padding='same', activation='tanh')(y4)

        return Model(inputs, outIm)

    def discriminator(self):
        def conv(inputs, filters, size=4, normalize=True):
            x = Conv2D(filters, kernel_size=size, strides=2, padding='same')(inputs)
            x = LeakyReLU(alpha=0.2)(x)
            if normalize:
                x = (InstanceNormalization())(x)
            return x

        inputs = Input(shape=self.shape)

        x1 = conv(inputs, self.g_filters)
        x2 = conv(x1, self.g_filters * 2)
        x3 = conv(x2, self.g_filters * 4)
        x4 = conv(x3, self.g_filters * 8)

        y = Conv2D(1, kernel_size=4, strides=1, padding='same')(x4)

        return Model(inputs, y)

    def train(self, epochs, batch_size=1, batches=0, prefix=""):


        start_time = datetime.datetime.now()

        real = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        print(real.shape)
        imageSet = ImageDataLoader(batch_size, res=(self.shape[0], self.shape[1]))
        drawingSet = DrawingDataLoader('flower', batch_size, res=(self.shape[0], self.shape[1]))
        if batches == 0:
            batches = drawingSet.N
        for epoch in range(epochs):
            for batch_i in range(batches):
                print("fetching data")
                imgs_x = drawingSet.get_batch()
                imgs_y = imageSet.get_batch()
                print("done fetching data")
                # Train Discriminators
                fake_y = self.g_XY(imgs_x, training=True)
                fake_x = self.g_YX(imgs_y, training=True)

                # Calculate losses
                dX_real_loss = self.d_X.train_on_batch(imgs_x, real)
                dX_fake_loss = self.d_X.train_on_batch(fake_x, fake)
                dX_loss = 0.5 * np.add(dX_real_loss, dX_fake_loss)

                dY_real_loss = self.d_Y.train_on_batch(imgs_y, real)
                dY_fake_loss = self.d_Y.train_on_batch(fake_y, fake)
                dY_loss = 0.5 * np.add(dY_real_loss, dY_fake_loss)

                d_loss = 0.5 + np.add(dX_loss, dY_loss)

                # Train Generators
                g_loss = self.combined.train_on_batch([imgs_x, imgs_y],
                                                      [real, real,
                                                       imgs_x, imgs_y, imgs_x, imgs_y])
                elapsed_time = datetime.datetime.now() - start_time

                # Plot progress
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
                    % (epoch, epochs,
                       batch_i, drawingSet.N,
                       d_loss[0], 100 * d_loss[1],
                       g_loss[0],
                       np.mean(g_loss[1:3]),
                       np.mean(g_loss[3:5]),
                       np.mean(g_loss[5:6]),
                       elapsed_time))

        self.d_X.save(prefix + "dx.h5")
        self.d_Y.save(prefix + "dy.h5")
        self.g_XY.save(prefix + "gy.h5")
        self.g_YX.save(prefix + "gx.h5")


if __name__ == '__main__':
    gan = CycleGan((64, 64, 3))
    gan.train(epochs=1, batch_size=1, batches=20, prefix='0')
    for i in range(99999):
        clear_session()
        tf.reset_default_graph()
        print("Starting stage %d" % i)
        gan = CycleGan((64, 64, 3), load=True, prefix="%d" % (i))
        gan.train(epochs=1, batch_size=1, batches=20,  prefix="%d" % (i+1))


