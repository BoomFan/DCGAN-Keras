#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
import os
import glob
from shutil import copyfile
import sys


# Dataset path(before splitting)
dataset_path_before = '../data/images/1OriginalImages_2Classes'
if not os.path.exists(dataset_path_before):
    print(dataset_path_before, 'does NOT exists!')
    sys.exit()
# Dataset path(after splitting)
dataset_path_after = '../data/images/4OriginalImages_2Classes_split'
if not os.path.exists(dataset_path_after):
    os.makedirs(dataset_path_after)
label_names = ['Class1', 'Class2']
# label_names = ['000ppm', '001ppm', '002ppm', '004ppm', '005ppm', '010ppm', '020ppm', '030ppm']
split_folder_names = ['train', 'val', 'test']

# Check if goal folders already exist
for i in range(len(label_names)):
    for j in range(len(split_folder_names)):
        creat_path = dataset_path_after + '/' + split_folder_names[j] + '/' + label_names[i]
        if not os.path.exists(creat_path):
            os.makedirs(creat_path)
        else:
            print('Splitted dataset', creat_path, 'already exists!')
            sys.exit()

# Get all the image names under all classes
file_name_list = []
label_name_list = []
for label_idx in range(len(label_names)):
    image_path = dataset_path_before + '/' + label_names[label_idx] + '/*.png'
    for img_full_path in glob.glob(image_path):
        base = os.path.basename(img_full_path)
        img_name = os.path.splitext(base)[0]
        file_name_list = file_name_list + [img_name]
        label_name_list = label_name_list + [label_names[label_idx]]
# print("file_name_list = ", file_name_list)
# print("label_name_list = ", label_name_list)

# Pick 20% of the total images to be the test set
file_name_train_val, file_name_test, label_name_train_val, label_name_test = train_test_split(file_name_list, label_name_list, test_size=0.2, random_state=1)

# Pick 25% of the train set images to be the val set
file_name_train, file_name_val, label_name_train, label_name_val = train_test_split(file_name_train_val, label_name_train_val, test_size=0.25, random_state=1)
print("lenth of file_name_train = ", len(file_name_train))
print("lenth of label_name_train = ", len(label_name_train))
print("lenth of file_name_val = ", len(file_name_val))
print("lenth of label_name_val = ", len(label_name_val))
print("lenth of file_name_test = ", len(file_name_test))
print("lenth of label_name_test = ", len(label_name_test))

# Create the splitted dataset
# Training set
for i in range(len(file_name_train)):
    print('Copying the ', i+1, '/', len(file_name_train), 'th file in training set')
    # find original files full path
    src_path = dataset_path_before + '/' + label_name_train[i] + '/' + file_name_train[i] + '.png'
    # make destination files full path
    dst_path = dataset_path_after + '/train/' + label_name_train[i] + '/' + file_name_train[i] + '.png'
    copyfile(src_path, dst_path)
# Val set
for i in range(len(file_name_val)):
    print('Copying the ', i+1, '/', len(file_name_val), 'th file in validation set')
    # find original files full path
    src_path = dataset_path_before + '/' + label_name_val[i] + '/' + file_name_val[i] + '.png'
    # make destination files full path
    dst_path = dataset_path_after + '/val/' + label_name_val[i] + '/' + file_name_val[i] + '.png'
    copyfile(src_path, dst_path)
# Test set
for i in range(len(file_name_test)):
    print('Copying the ', i+1, '/', len(file_name_val), 'th file in test set')
    # find original files full path
    src_path = dataset_path_before + '/' + label_name_test[i] + '/' + file_name_test[i] + '.png'
    # make destination files full path
    dst_path = dataset_path_after + '/test/' + label_name_test[i] + '/' + file_name_test[i] + '.png'
    copyfile(src_path, dst_path)
