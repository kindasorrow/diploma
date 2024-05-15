import os
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import warnings

from math import *
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from skimage.draw import polygon, polygon_perimeter
from skimage import measure
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.morphology import dilation, disk
from skimage.draw import polygon_perimeter
from datetime import date


class UNet:
    classes = 3
    sample_size = (256, 256)
    output_size = (1280, 720)
    weights_date = "2024-05-15"
    images = sorted(glob.glob("C:/Work/navigation/Размеченное/img/*.jpg"))
    masks = sorted(glob.glob("C:/Work/navigation/Размеченное/no_ceil_masks/*.png"))

    train_dataset = None
    test_dataset = None
    model = None

    def __init__(self):
        pass

        """
        Loads and processes images and masks for the UNet model.

        Parameters:
            image (str): The file path to the image.
            mask (str): The file path to the mask.

        Returns:
            tuple: A tuple containing the processed image and masks.
        """

    def __load_images(self, image, mask):
        image = tf.io.read_file(image)
        image = tf.io.decode_jpeg(image)
        image = tf.image.resize(image, self.output_size)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = image / 255.0

        mask = tf.io.read_file(mask)
        mask = tf.io.decode_png(mask)
        mask = tf.broadcast_to(mask, (1280, 720, 3))  # нужно будет унифицировать
        mask = tf.image.rgb_to_grayscale(mask)
        mask = tf.image.resize(mask, self.output_size)
        mask = tf.image.convert_image_dtype(mask, tf.float32)

        masks = []

        for i in range(self.classes):
            masks.append(tf.where(tf.equal(mask, float(i)), 1.0, 0.0))

        masks = tf.stack(masks, axis=2)
        masks = tf.reshape(masks, self.output_size + (self.classes,))

        return image, masks

    def __augmentate_images(self, image, masks):
        random_crop = tf.random.uniform((), 0.3, 1)
        #   image = tf.image.central_crop(image, random_crop)  Данный блок обрезает маски но улучшает сегментацию
        #   masks = tf.image.central_crop(masks, random_crop)

        random_flip = tf.random.uniform((), 0, 1)
        if random_flip >= 0.5:
            image = tf.image.flip_left_right(image)
            masks = tf.image.flip_left_right(masks)

        image = tf.image.resize(image, self.output_size)
        masks = tf.image.resize(masks, self.output_size)

        return image, masks

    def __input_layer(self):
        return tf.keras.layers.Input(shape=self.sample_size + (3,))

    def __downsample_block(self, filters, size, batch_norm=True):
        initializer = tf.keras.initializers.GlorotNormal()

        result = tf.keras.Sequential()

        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                   kernel_initializer=initializer, use_bias=False))

        if batch_norm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())
        return result

    def __upsample_block(self, filters, size, dropout=False):
        initializer = tf.keras.initializers.GlorotNormal()

        result = tf.keras.Sequential()

        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                            kernel_initializer=initializer, use_bias=False))

        result.add(tf.keras.layers.BatchNormalization())

        if dropout:
            result.add(tf.keras.layers.Dropout(0.25))

        result.add(tf.keras.layers.ReLU())
        return result

    def __output_layer(self, size):
        initializer = tf.keras.initializers.GlorotNormal()
        return tf.keras.layers.Conv2DTranspose(self.classes, size, strides=2, padding='same',
                                               kernel_initializer=initializer, activation='sigmoid')

    def __dice_mc_metric(self, a, b):
        a = tf.unstack(a, axis=3)
        b = tf.unstack(b, axis=3)

        dice_summ = 0

        for i, (aa, bb) in enumerate(zip(a, b)):
            numenator = 2 * tf.math.reduce_sum(aa * bb) + 1
            denomerator = tf.math.reduce_sum(aa + bb) + 1
            dice_summ += numenator / denomerator

        avg_dice = dice_summ / self.classes

        return avg_dice

    def __dice_mc_loss(self, a, b):
        return 1 - self.__dice_mc_metric(a, b)

    def __dice_bce_mc_loss(self, a, b):
        return 0.3 * self.__dice_mc_loss(a, b) + tf.keras.losses.binary_crossentropy(a, b)

    def build_model(self):
        """
        Builds the UNet model architecture, compiles it, and sets it as the model attribute.
        """
        # dataset
        images_dataset = tf.data.Dataset.from_tensor_slices(self.images)
        masks_dataset = tf.data.Dataset.from_tensor_slices(self.masks)
        dataset = tf.data.Dataset.zip((images_dataset, masks_dataset))
        dataset = dataset.map(self.__load_images, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.repeat(60)
        dataset = dataset.map(self.__augmentate_images, num_parallel_calls=tf.data.AUTOTUNE)

        self.train_dataset = dataset.take(2000).cache()
        self.test_dataset = dataset.skip(2000).take(100).cache()
        self.train_dataset = self.train_dataset.batch(8)
        self.test_dataset = self.test_dataset.batch(8)

        inp_layer = self.__input_layer()

        downsample_stack = [
            self.__downsample_block(64, 4, batch_norm=False),
            self.__downsample_block(128, 4),
            self.__downsample_block(256, 4),
            self.__downsample_block(512, 4),
            self.__downsample_block(512, 4),
            self.__downsample_block(512, 4),
            self.__downsample_block(512, 4),
        ]

        upsample_stack = [
            self.__upsample_block(512, 4, dropout=True),
            self.__upsample_block(512, 4, dropout=True),
            self.__upsample_block(512, 4, dropout=True),
            self.__upsample_block(256, 4),
            self.__upsample_block(128, 4),
            self.__upsample_block(64, 4)
        ]

        out_layer = self.__output_layer(4)
        # Реализуем skip connections
        x = inp_layer

        downsample_skips = []

        for block in downsample_stack:
            x = block(x)
            downsample_skips.append(x)

        downsample_skips = reversed(downsample_skips[:-1])

        for up_block, down_block in zip(upsample_stack, downsample_skips):
            x = up_block(x)
            x = tf.keras.layers.Concatenate()([x, down_block])

        out_layer = out_layer(x)

        self.model = tf.keras.Model(inputs=inp_layer, outputs=out_layer)
        self.model.compile(optimizer='adam', loss=[self.__dice_bce_mc_loss], metrics=[self.__dice_mc_metric])

    def train(self, epochs=17, initial_epoch=0):
        """
        Trains the model for a specified number of epochs.

        Parameters:
            epochs (int): The number of epochs to train the model (default is 17).
            initial_epoch (int): The initial epoch to start training from (default is 0).

        Returns:
            dict: A dictionary containing the training history.
        """
        history_dice = self.model.fit(self.train_dataset, validation_data=self.test_dataset, epochs=epochs,
                                      initial_epoch=initial_epoch)
        self.model.save_weights('C:\\Work\\navigation\\wights\\' + str(date.today()) + '.weights.h5')
        return history_dice.history

    def load_weights(self):
        """
        Loads the weights for the model based on the weights date.
        """
        self.model.load_weights('C:\\Work\\navigation\\wights\\' + self.weights_date + '.weights.h5')

    def predict(self, image_path):
        """
        Predicts the output based on the input image path.

        Parameters:
            image_path (str): The file path to the input image.

        Returns:
            numpy array: The predicted output.
        """
        frame = cv2.imread(image_path)
        sample = resize(frame, self.sample_size)
        predict = self.model.predict(sample.reshape((1,) + self.sample_size + (3,)))
        predict = predict.reshape(self.sample_size + (self.classes,))
        return predict
