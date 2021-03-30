import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow.keras import layers, Input
import time
'''
Rescale Factor is the parameter used to reduce the quality of celeb face image 
This code generates image of size (256,256, 3)
'''
class Generator:
    def __init__(self):
        self.input = Input(shape = ([64, 64, 3]))
        self.res_blocks = 8


    def Conv_Block(self, filters, apply_batchnorm = True):
        initializer = tf.random_normal_initializer(0., 0.09)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, kernel_size = 3, strides=1, padding='same',
                                    kernel_initializer=initializer, use_bias=False))

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result

    def upsample(self, filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))

        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result


    def Generator(self):
        initializer = tf.random_normal_initializer(0., 0.09)
        last_layer = tf.keras.layers.Conv2DTranspose(3, kernel_size = 3, strides = 1, padding = 'same', kernel_initializer = initializer, activation = 'tanh')
        first = self.input
        res_blocks = [
            self.Conv_Block(64),  # (bs,32, 32, 64)
            self.Conv_Block(64),  # (bs,32, 32, 64)
            self.Conv_Block(128),  # (bs, 32, 32, 128)
            self.Conv_Block(128),  # (bs, 32, 32, 128)
            self.Conv_Block(256),  # (bs, 32, 32, 256)
            self.Conv_Block(256),  # (bs, 32, 32, 256)
            self.Conv_Block(512),  # (bs, 32, 32, 512)
            self.Conv_Block(512),  # (bs, 32, 32, 512)
            self.Conv_Block(256),  # (bs, 32, 32, 512)
            self.Conv_Block(256),  # (bs, 32, 32, 512)
            self.Conv_Block(128),  # (bs, 32, 32, 128)
            self.Conv_Block(128),  # (bs, 32, 32, 128)
        ]

        x = self.Conv_Block(64, apply_batchnorm = False)(first)
        t = x
        lay = []
        for i  in range(0, len(res_blocks), 2):
            t = res_blocks[i](x)
            lay.append(t)
            t = res_blocks[i+1](t)
            t = layers.Add()([t, lay[-1]])

        upsample_inp = self.upsample(16, 4)
        t = upsample_inp(t)
        x = last_layer(t)
        return tf.keras.Model(inputs = self.input, outputs = x)


class Discriminator:
    def __init__(self):
        self.input = Input(shape = ([256, 256, 3]))

    def Conv_2D(self, filters):
        initializer = tf.random_normal_initializer(0., 0.09)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, kernel_size = 3, strides=1, padding='same',
                                    kernel_initializer=initializer, use_bias=False))

        result.add(
            tf.keras.layers.MaxPool2D(2,2)
        )

        result.add(tf.keras.layers.LeakyReLU())

        return result

    def Discriminator(self):
        d = self.Conv_2D(64)(self.input)
        d1 = self.Conv_2D(128)(d)
        d2 = self.Conv_2D(64)(d1)
        d3 = self.Conv_2D(32)(d2)
        d4 = self.Conv_2D(16)(d3)
        d5 = tf.keras.layers.Dense(10)(d4)
        d6 = tf.keras.layers.LeakyReLU(alpha =0.2)(d5)
        final = tf.keras.layers.Dense(1, activation='sigmoid')(d6)
        return tf.keras.Model(inputs = self.input, outputs = final)