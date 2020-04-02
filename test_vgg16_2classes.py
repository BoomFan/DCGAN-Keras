import os
from keras import models
from keras.preprocessing.image import ImageDataGenerator
import sys
import json
import codecs


def saveHist(path, history):  # Used to save trining history
    with codecs.open(path, 'w', encoding='utf-8') as f:
        json.dump(history, f, separators=(',', ':'), sort_keys=True, indent=4)


# The directory where we store our augmented dataset
base_dir = '../data/images/5AugmentedImages_2Classes_split'
if not os.path.exists(base_dir):
    print(base_dir, 'does NOT exists!')
    sys.exit()

# Set parameters
img_height = 200
img_width = 200
# train_sample_num = 620
# val_sample_num = 210
test_sample_num = 210
batch_size = 10

output_dir = '../data/output/VGG16_2classes_all_new'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
model_filename = output_dir + '/foodsafety_VGG16_2classes.h5'
test_result_filename = output_dir + '/test_result.json'
# Directories for testing splits
test_dir = os.path.join(base_dir, 'test')

# Generate data inputs for training, validation and testing
datagen = ImageDataGenerator(rescale=1./255)

# Construct model layers
if os.path.exists(model_filename):
    # load model is a saved model already exists
    model = models.load_model(model_filename)
    # summarize model.
    model.summary()
else:
    print('Model Not Found!')
    sys.exit()

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

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
