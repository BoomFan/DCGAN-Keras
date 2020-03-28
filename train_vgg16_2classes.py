import os
import json
import codecs
import sys
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
# This is module with image preprocessing utilities
from keras.applications import VGG16
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
import numpy as np


class LossHistory(Callback):
    # https://stackoverflow.com/a/53653154/852795
    def on_epoch_end(self, epoch, logs=None):
        new_history = {}
        for k, v in logs.items():  # compile new history from logs
            new_history[k] = [v]  # convert values into lists
        current_history = loadHist(history_filename)  # load history from current training
        current_history = appendHist(current_history, new_history)  # append the logs
        saveHist(history_filename, current_history)  # save history from current training


def saveHist(path, history):  # Used to save trining history
    with codecs.open(path, 'w', encoding='utf-8') as f:
        json.dump(history, f, separators=(',', ':'), sort_keys=True, indent=4)


def loadHist(path):  # Used to load trining history
    n = {}  # set history to empty
    if os.path.exists(path):  # reload history if it exists
        with codecs.open(path, 'r', encoding='utf-8') as f:
            n = json.loads(f.read())
    return n


def appendHist(h1, h2):
    if h1 == {}:
        return h2
    else:
        dest = {}
        for key, value in h1.items():
            dest[key] = value + h2[key]
        return dest


# def extract_features(directory, sample_count):
#     # features is the last layer
#     # since we have 5 MaxPooling2D, the last layer shape should be (None, 6, 6, 512)
#     features = np.zeros(shape=(sample_count, 6, 6, 512))
#     labels = np.zeros(shape=(sample_count))
#     generator = datagen.flow_from_directory(
#         directory,
#         target_size=(img_height, img_width),
#         batch_size=batch_size,
#         class_mode='binary')
#     i = 0
#     for inputs_batch, labels_batch in generator:
#         # print('i = ', i)
#         features_batch = conv_base.predict(inputs_batch)
#         # print('inputs_batch.shape = ', inputs_batch.shape)  # (20, 200, 200, 3)
#         features[i*batch_size: (i + 1)*batch_size] = features_batch
#         labels[i*batch_size: (i + 1)*batch_size] = labels_batch
#         i += 1
#         if i * batch_size >= sample_count:
#             # Note that since generators yield data indefinitely in a loop,
#             # we must `break` after every image has been seen once.
#             break
#     return features, labels


def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


# The directory where we store our augmented dataset
# (we still do data augmentation during the training)
base_dir = '../data/images/5AugmentedImages_2Classes_split'
if not os.path.exists(base_dir):
    print(base_dir, 'does NOT exists!')
    sys.exit()

# Set parameters
img_height = 200
img_width = 200
train_sample_num = 624
val_sample_num = 208
test_sample_num = 208
batch_size = 20
steps_per_epoch = train_sample_num//batch_size + 1
epochs = 100
smooth_factor = 0  # For plotting the accuracy and loss

output_dir = '../data/output/VGG16_2classes_all_trainable'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
model_filename = output_dir + '/foodsafety_VGG16_2classes.h5'
history_filename = output_dir + '/model-history.json'
test_result_filename = output_dir + '/test_result.json'

# Directories for our training, validation and testing splits
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# VGG16 has 13 Convolutional Layer and 3 Fully connected Layer. Thus 16=13+3
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(img_height, img_width, 3))

conv_base.summary()

# # Generate data inputs for training, validation and testing
# datagen = ImageDataGenerator(rescale=1./255)
#
# train_features, train_labels = extract_features(train_dir, train_sample_num)
# print('train_features.shape = ', train_features.shape)
# print('train_labels.shape = ', train_labels.shape)
# validation_features, validation_labels = extract_features(validation_dir, val_sample_num)
# print('validation_features.shape = ', validation_features.shape)
# print('validation_labels.shape = ', validation_labels.shape)
# test_features, test_labels = extract_features(test_dir, test_sample_num)
# print('test_features.shape = ', test_features.shape)
# print('test_labels.shape = ', test_labels.shape)
#
# train_features = np.reshape(train_features, (train_sample_num, 6 * 6 * 512))
# validation_features = np.reshape(validation_features, (val_sample_num, 6 * 6 * 512))
# test_features = np.reshape(test_features, (test_sample_num, 6 * 6 * 512))

# Construct model layers
if os.path.exists(model_filename):
    # load model is a saved model already exists
    model = models.load_model(model_filename)
    # summarize model.
    model.summary()
else:
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.summary()

print('This is the number of trainable weights '
      'before conv_base.trainable = True:', len(model.trainable_weights))
conv_base.trainable = True

# # if we want to finetune some layers
# # set the layers before block5_conv1 to untrainable
# # set the layers equal or after block5_conv1 to trainable
# set_trainable = False
# for layer in conv_base.layers:
#     print('layer.name = ', layer.name)
#     if layer.name == 'block5_conv1':
#         set_trainable = True
#     if set_trainable:
#         layer.trainable = True
#     else:
#         layer.trainable = False
# print('This is the number of trainable weights '
#       'after conv_base.trainable = True:', len(model.trainable_weights))

# Augment training data.
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=180,
      # width_shift_range=0.2,
      # height_shift_range=0.2,
      # shear_range=0.2,
      # zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to img_height x img_width
        target_size=(img_height, img_width),
        batch_size=batch_size,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

# Compile model
print('Compile model')
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr=1e-5),
              # optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])

# Create checkpoints
print('Create checkpoints')
model_checkpoint = ModelCheckpoint(model_filename, verbose=1, period=1, save_best_only=True, mode='min')
history_checkpoint = LossHistory()
callbacks_list = [model_checkpoint, history_checkpoint]

print('Now training starts!')
vgg16_history = model.fit_generator(
      train_generator,
      steps_per_epoch=steps_per_epoch,
      epochs=epochs,
      validation_data=validation_generator,
      validation_steps=50,
      callbacks=callbacks_list)
#
# print('appendHist ...')
# history = appendHist(history, new_history.history)

# Plot accuracy and loss
historical_history = loadHist(history_filename)  # load history from current training
print('Plot accuracy and loss ...')
# acc = vgg16_history.history['acc']
# val_acc = vgg16_history.history['val_acc']
# loss = vgg16_history.history['loss']
# val_loss = vgg16_history.history['val_loss']
acc = historical_history['acc']
val_acc = historical_history['val_acc']
loss = historical_history['loss']
val_loss = historical_history['val_loss']

epochs = range(len(acc))

plt.figure()
plt.plot(epochs, acc, 'b-', label='Training acc')
plt.plot(epochs, acc, 'b*', label='Training acc')
plt.plot(epochs, val_acc, 'r--', label='Validation acc')
plt.plot(epochs, val_acc, 'ro', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
# plt.show()
image_name = output_dir + '/accuracy_smooth_'+str(smooth_factor)+'.png'
plt.savefig(image_name)
plt.close()


plt.figure()
plt.plot(epochs, loss, 'b-', label='Training loss')
plt.plot(epochs, loss, 'b*', label='Training loss')
plt.plot(epochs, val_loss, 'r--', label='Validation loss')
plt.plot(epochs, val_loss, 'ro', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

image_name = output_dir + '/loss_smooth_'+str(smooth_factor)+'.png'
plt.savefig(image_name)
plt.close()


test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=test_sample_num//batch_size)
print('test loss:', test_loss)
print('test acc:', test_acc)
test_result = {}
test_result['test acc'] = [test_acc]
test_result['test loss'] = [test_loss]
saveHist(test_result_filename, test_result)
