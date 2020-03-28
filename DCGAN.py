from keras.models import Sequential, Model, load_model
from keras.layers import UpSampling2D, Conv2D, Activation, BatchNormalization, Reshape, Dense, Input, LeakyReLU, Dropout, Flatten, ZeroPadding2D
from keras.optimizers import Adam

import glob
from PIL import Image
import numpy as np
import os
import argparse
from ast import literal_eval

from scipy.misc import imsave


class DCGAN:
    def __init__(self, discriminator_path, generator_path, epoch_params_path, output_directory, img_size):
        self.img_size = img_size
        self.upsample_layers = 5
        self.starting_filters = 64
        self.kernel_size = 3
        self.channels = 3
        self.epoch_init_num = 0
        self.discriminator_path = discriminator_path
        self.generator_path = generator_path
        self.epoch_params_path = epoch_params_path
        self.output_directory = output_directory

    def build_generator(self):
        noise_shape = (100,)

        # This block of code can be a little daunting, but essentially it automatically calculates the required starting
        # array size that will be correctly upscaled to our desired image size.
        #
        # We have 5 Upsample2D layers which each double the images width and height, so we can determine the starting
        # x size by taking (x / 2^upsample_count) So for our target image size, 256x192, we do the following:
        # x = (192 / 2^5), y = (256 / 2^5) [x and y are reversed within the model]
        # We also need a 3rd dimension which is chosen relatively arbitrarily, in this case it's 64.
        model = Sequential()
        model.add(
            Dense(self.starting_filters * (self.img_size[0] // (2 ** self.upsample_layers))  *  (self.img_size[1] // (2 ** self.upsample_layers)),
                  activation="relu", input_shape=noise_shape))
        model.add(Reshape(((self.img_size[0] // (2 ** self.upsample_layers)),
                           (self.img_size[1] // (2 ** self.upsample_layers)),
                           self.starting_filters)))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())  # 6x8 -> 12x16
        model.add(Conv2D(1024, kernel_size=self.kernel_size, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())  # 12x16 -> 24x32
        model.add(Conv2D(512, kernel_size=self.kernel_size, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())  # 24x32 -> 48x64
        model.add(Conv2D(256, kernel_size=self.kernel_size, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())  # 48x64 -> 96x128
        model.add(Conv2D(128, kernel_size=self.kernel_size, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())  # 96x128 -> 192x256
        model.add(Conv2D(64, kernel_size=self.kernel_size, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(32, kernel_size=self.kernel_size, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(self.channels, kernel_size=self.kernel_size, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        img_shape = (self.img_size[0], self.img_size[1], self.channels)

        model = Sequential()

        model.add(Conv2D(32, kernel_size=self.kernel_size, strides=2, input_shape=img_shape, padding="same"))  # 192x256 -> 96x128
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, kernel_size=self.kernel_size, strides=2, padding="same"))  # 96x128 -> 48x64
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(128, kernel_size=self.kernel_size, strides=2, padding="same"))  # 48x64 -> 24x32
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(256, kernel_size=self.kernel_size, strides=1, padding="same"))  # 24x32 -> 12x16
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(512, kernel_size=self.kernel_size, strides=1, padding="same"))  # 12x16 -> 6x8
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)

    def build_gan(self):
        optimizer = Adam(0.0002, 0.5)

        # See if the specified model paths exist, if they don't then we start training new models

        if os.path.exists(self.discriminator_path) and os.path.exists(self.generator_path):
            self.discriminator = load_model(self.discriminator_path)
            self.generator = load_model(self.generator_path)
            print("Loaded models...")
        else:
            self.discriminator = self.build_discriminator()
            self.discriminator.compile(loss='binary_crossentropy',
                                       optimizer=optimizer,
                                       metrics=['accuracy'])

            self.generator = self.build_generator()
            self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)
        if os.path.exists(self.epoch_params_path):
            self.epoch_init_num = np.load(self.epoch_params_path)
            print('Loading from epoch ', self.epoch_init_num, '...')

        # These next few lines setup the training for the GAN model
        z = Input(shape=(100,))
        img = self.generator(z)

        self.discriminator.trainable = False

        valid = self.discriminator(img)

        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def load_imgs(self, image_path, image_size=None):
        X_train = []
        # print("load_imgs: folder_path = ", folder_path)
        # image_path = folder_path + "/*.png"
        print("load_imgs: image_path = ", image_path)
        # print("load_imgs: glob.glob(image_path) = ", glob.glob(image_path))
        for i in glob.glob(image_path):
            # print("i = ", i)
            img = Image.open(i)
            if image_size is not None:
                # print('image_size = ', image_size)
                img = img.resize(image_size)
            # print("img = ", img)
            img = np.asarray(img)
            # print("img.shape = ", img.shape)
            # if X_train is not None:
            #     X_train = img
            # else:
            #     X_train = np.concatenate((X_train, img), axis=4)
            X_train.append(img)
        X_train = np.stack(X_train, axis=0)
        # print("X_train.shape = ", X_train.shape)
        return np.asarray(X_train)
        # return X_train

    def train(self, epochs, image_path, batch_size=32, image_size=None, save_interval=50):
        self.build_gan()
        # print("image_path = ", image_path)
        X_train = self.load_imgs(image_path, image_size)
        # print("X_train = ", X_train)
        # print("Training Data Shape: ", X_train.shape)

        # Rescale images from -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5

        half_batch = batch_size // 2

        start_epoch_num = 0
        if self.epoch_init_num != 0:
            start_epoch_num = self.epoch_init_num + 1
        for epoch in range(start_epoch_num, epochs):

            # Train Generator
            noise = np.random.normal(0, 1, (batch_size, 100))
            g_loss = self.combined.train_on_batch(noise, np.ones((batch_size, 1)))

            # Train Discriminator
            # print("X_train.shape[0] = ", X_train.shape[0])
            # print("half_batch = ", half_batch)
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            # Sample noise and generate a half batch of new images
            noise = np.random.normal(0, 1, (half_batch, 100))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Print progress
            print(epoch, '[D loss:', d_loss[0], '| D Accuracy: ', 100 * d_loss[1], '] [G loss: ', g_loss, ']')

            # If at save interval => save generated image samples, save model files
            if epoch % (save_interval) == 0:
                self.save_imgs(epoch)
                save_path = self.output_directory + "/models"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                self.discriminator.save(save_path + "/discrim.h5")
                self.generator.save(save_path + "/generat.h5")
                # Save the current epoch number
                np.save(save_path + "/epoch_params", epoch)

    def gene_imgs(self, count):
        # Generate images from the currently loaded model
        noise = np.random.normal(0, 1, (count, 100))
        return self.generator.predict(noise)

    def save_imgs(self, epoch):
        r, c = 5, 5

        # Generates r*c images from the model, saves them individually and as a gallery

        imgs = self.gene_imgs(r*c)
        imgs = 0.5 * imgs + 0.5

        for i, img_array in enumerate(imgs):
            path = self.output_directory + "/generated_" + str(self.img_size[0]) + "x" + str(self.img_size[1])
            if not os.path.exists(path):
                os.makedirs(path)
            imsave(path + "/epoch_" + str(epoch) + "_" + str(i) + ".png", img_array)

        nindex, height, width, intensity = imgs.shape
        nrows = nindex // c
        assert nindex == nrows * c
        # want result.shape = (height*nrows, width*ncols, intensity)
        gallery = (imgs.reshape(nrows, c, height, width, intensity)
                   .swapaxes(1, 2)
                   .reshape(height * nrows, width * c, intensity))

        path = self.output_directory + "/gallery_generated_" + str(self.img_size[0]) + "x" + str(self.img_size[1])
        if not os.path.exists(path):
            os.makedirs(path)
        imsave(path + "/" + str(epoch) + ".png", gallery)

    def generate_imgs(self, count, threshold, modifier):
        self.build_gan()

        # Generates (count) images from the model ensuring the discriminator scores them between the threshold values
        # and saves them

        imgs = []
        for i in range(count):
            score = [0]
            while not(threshold[0] < score[0] < threshold[1]):
                img = self.gene_imgs(1)
                score = self.discriminator.predict(img)
            print("Image found: ", score[0])
            imgs.append(img)

        imgs = np.asarray(imgs).squeeze()
        imgs = 0.5 * imgs + 0.5

        print(imgs.shape)
        for i, img_array in enumerate(imgs):
            path = self.output_directory + "/generated_" + str(threshold[0]) + "_" + str(threshold[1])
            if not os.path.exists(path):
                os.makedirs(path)
            imsave(path + "/" + str(modifier) + "_" + str(i) + ".png", img_array)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--load_generator',
                        help='Path to existing generator weights file',
                        default="../data/output/test_3AugmentedImages/Class1/models/generat.h5")
    parser.add_argument('--load_discriminator',
                        help='Path to existing discriminator weights file',
                        default="../data/output/test_3AugmentedImages/Class1/models/discrim.h5")
    parser.add_argument('--load_epoch_params',
                        help='Path to existing epoch parameters file',
                        default="../data/output/test_3AugmentedImages/Class1/models/epoch_params.npy")

    parser.add_argument('--data',
                        help='Path to directory of images of correct dimensions, using *.[filetype] (e.g. *.png) to reference images',
                        default="../data/images/3AugmentedImages/Class1/*.png")
    parser.add_argument('--sample',
                        help='If given, will generate that many samples from existing model instead of training',
                        default=-1)
    parser.add_argument('--sample_thresholds',
                        help='The values between which a generated image must score from the discriminator',
                        default="(0.0, 0.1)")
    # parser.add_argument('--batch_size', help='Number of images to train on at once', default=24)
    parser.add_argument('--batch_size',
                        help='Number of images to train on at once',
                        default=16)
    # parser.add_argument('--image_size', help='Size of images as tuple (height,width). Height and width must both be divisible by (2^5)', default="(192, 256)")
    parser.add_argument('--image_size',
                        help='Size of images as tuple (height,width). Height and width must both be divisible by (2^5)',
                        default="(128, 128)")
    parser.add_argument('--epochs',
                        help='Number of epochs to train for',
                        default=100000)
    parser.add_argument('--save_interval',
                        help='How many epochs to go between saves/outputs',
                        default=200)
    parser.add_argument('--output_directory',
                        help="Directoy to save weights and images to.",
                        default="../data/output/test_3AugmentedImages/Class1")

    args = parser.parse_args()

    dcgan = DCGAN(args.load_discriminator,
                  args.load_generator,
                  args.load_epoch_params,
                  args.output_directory,
                  literal_eval(args.image_size))
    if args.sample == -1:
        dcgan.train(epochs=int(args.epochs),
                    image_path=args.data,
                    batch_size=int(args.batch_size),
                    image_size=literal_eval(args.image_size),
                    save_interval=int(args.save_interval))
    else:
        dcgan.generate_imgs(int(args.sample),
                            literal_eval(args.sample_thresholds),
                            "")
