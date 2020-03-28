import os
import sys
from keras.preprocessing.image import ImageDataGenerator

# This is module with image preprocessing utilities
from keras.preprocessing import image
import numpy as np


# The directory where we will
# store our augmented dataset
origin_data_dir = '../data/images/4OriginalImages_2Classes_split'
if not os.path.exists(origin_data_dir):
    print(origin_data_dir, 'does NOT exists!')
    sys.exit()

augmented_data_dir = '../data/images/5AugmentedImages_2Classes_split'
if not os.path.exists(augmented_data_dir):
    os.mkdir(augmented_data_dir)
else:
    print(augmented_data_dir, 'already exists!')
    sys.exit()

split_folder_names = ['train', 'val', 'test']
class_name_list = ['Class1', 'Class2']
# class_name_list = ['000ppm', '001ppm', '002ppm', '004ppm', '005ppm', '010ppm', '020ppm', '030ppm']

# The first for-loop is for our training, validation and test splits
for split_name in split_folder_names:
    origin_data_split = os.path.join(origin_data_dir, split_name)
    if not os.path.exists(origin_data_split):
        os.mkdir(origin_data_split)
    augmented_data_split = os.path.join(augmented_data_dir, split_name)
    if not os.path.exists(augmented_data_split):
        os.mkdir(augmented_data_split)
    # Now to data sugmentation for each class
    for class_name in class_name_list:
        origin_class_dir = os.path.join(origin_data_split, class_name)
        augmented_class_dir = os.path.join(augmented_data_split, class_name)
        if not os.path.exists(augmented_class_dir):
            os.mkdir(augmented_class_dir)

        # Using data augmentation
        datagen = ImageDataGenerator(rotation_range=180,
                                     width_shift_range=0,
                                     height_shift_range=0,
                                     shear_range=0,
                                     zoom_range=0,
                                     horizontal_flip=True,
                                     fill_mode='nearest')

        # Get the file name of original images
        fnames = [os.path.join(origin_class_dir, fname) for fname in os.listdir(origin_class_dir)]
        # print("fnames = ", fnames)
        # A typical image name may looks like this:
        # a01245_001_02_3_o.png
        # a01245: This image comes from group "a", which is a combination of 0-5 ppm.
        # 001: The exact ppm of this image is 001ppm
        # 02: The second sample within group 001
        # 3: This image shows the 03rd pad among all four pads.(top left, top right, bottom left, bottom right)
        # o: Original image

        # Stack images into one Numpy array and resize it
        img_batch = None
        for i in range(len(fnames)):
            # Read the image and resize it
            img_path = fnames[i]
            img = image.load_img(img_path, target_size=(200, 200))
            # Convert it to a Numpy array with shape (200, 200, 3)
            x = image.img_to_array(img)
            # Reshape it to (1, 200, 200, 3)
            x = x.reshape((1,) + x.shape)
            if img_batch is None:
                img_batch = x
                # print("img_batch.shape = ", img_batch.shape)
                # print("x.shape = ", x.shape)
            else:
                img_batch = np.concatenate((img_batch, x), axis=0)
                # print("img_batch.shape = ", img_batch.shape)
                # print("x.shape = ", x.shape)

        num_loops = 10
        for ind in range(num_loops):
            # The .flow() command below generates batches of randomly transformed images.
            # It will loop indefinitely, so we need to `break` the loop at some point!
            i = 0

            # We don't use the "save_to_dir" argument, because datagen.flow() cannot assign
            # different prefix for different images
            # for batch in datagen.flow(img_batch,
            #                           batch_size=1,
            #                           shuffle=False,
            #                           save_to_dir=augmented_class_dir):
            #     # Make image name
            #     image_name = 'augmented_'+str(i)+'.png'
            #     image_path_name = os.path.join(augmented_class_dir, image_name)
            #     print("image_path_name = ", image_path_name)
            for batch in datagen.flow(img_batch, batch_size=1, shuffle=False):
                print("Loop number ", ind, ", i = ", i)
                # print("batch.shape = ", batch.shape)
                image_full_name = os.path.basename(fnames[i])
                image_name = os.path.splitext(image_full_name)[0]
                # print("image_name = ", image_name)
                image_name = image_name+"_r_f_"+str(ind).zfill(2)+".png"
                # A typical image name may looks like this:
                # a01245_001_02_3_o_r_f_04.png
                # a01245: This image comes from group "a", which is a combination of 0-5 ppm.
                # 001: The exact ppm of this image is 001ppm
                # 02: The second sample within group 001
                # 03: This image shows the 03rd pad among all four pads.(top left, top right, bottom left, bottom right)
                # o: Original image
                # r: Rotated
                # f: Flipped
                image_path_name = os.path.join(augmented_class_dir, image_name)
                # print("image_path_name = ", image_path_name)
                im = image.array_to_img(batch[0])
                im.save(image_path_name)

                i += 1
                if i == len(fnames):
                    break
