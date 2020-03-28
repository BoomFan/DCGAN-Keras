# Overall

This repository is created for Food Safety Project for prediction of heavy metals concentrations in food or water.

We currently tried two algorithms for detection: VGG16 and DCGAN.

Before everything starts, we should first build our dataset. Because our original dataset is small, we should first split it into  `train`, `val`, and `test`, then we can do data augmentation on these split images.

## Split dataset
Change the dataset path in the `split_dataset.py` file and then:
```
python3 split_dataset.py
```

## Data augmentation
To run data augmentation on the original dataset, please change the path names and the classes names in the `data_augmentation.py` file and then:
 ```
 python3 data_augmentation.py
 ```


# VGG16
In this section, we tried two different datasets. The first one labeled all images in two classes: ['Class1', 'Class2'] (0~5ppm, 10~30ppm). The second one labeled all images in eight classes: ['000ppm', '001ppm', '002ppm', '004ppm', '005ppm', '010ppm', '020ppm', '030ppm']

## VGG16 in two classes

### Training
To train a model with VGG16 structure, please check all parameters and the `output_dir` in the `train_vgg16_2classes.py` file. If everything is OK, you may run:
```
python3 train_vgg16_2classes.py
```
### Plot loss and accuracy
After the training finished, there should be images showing the accuracy and the loss inside the output directory.

However, if you want to rerun the plotting code, you may do this:
```
python3 plot_vgg16_2classes.py
```

### Testing
To test the model with our test set:
```
python3 test_vgg16_2classes.py
```

# DCGAN-Keras

## Introduction

This is a relatively simple Deep Convolutional Generative Adversarial Network built in Keras. Given a dataset of images it will be able
to generate new images similar to those in the dataset. It was originally built to generate landscape paintings such
as the ones shown below. As a result, also contained are some scripts for collecting artwork from ArtUK and resizing images to make them work with the network. There are also examples of it being trained on Space imagery as well.

## Model Architecture

Visualisations of the model architecture:

* [Generator](https://netbrix.co/#!/model/16b7385e-443f-11ea-9bd4-4e67a2099c39)
* [Discriminator](https://netbrix.co/#!/model/4a644677-443f-11ea-9bd4-4e67a2099c39)
* [Both (DCGAN)](https://netbrix.co/#!/model/e442d8b3-443e-11ea-9bd4-4e67a2099c39)

## Example Outputs

### Landscapes

The following images were generated at 256x192, then upscaled using [the bigjpg tool](https://bigjpg.com/) which is a GAN based upscaling tool.

|Mountain Lake|Peninsula|
|:-----------:|:-----------:|
| <img src="https://raw.githubusercontent.com/DataSnaek/DCGAN-Keras/master/data/output/upscaled/Mountain_Lake.png"> | <img src="https://raw.githubusercontent.com/DataSnaek/DCGAN-Keras/master/data/output/upscaled/Peninsula.png"> |

|Hill Bushes|Grassy Mountain|
|:-----------:|:-----------:|
| <img src="https://raw.githubusercontent.com/DataSnaek/DCGAN-Keras/master/data/output/upscaled/Hill_Bushes.png"> | <img src="https://raw.githubusercontent.com/DataSnaek/DCGAN-Keras/master/data/output/upscaled/Grassy_Mountain.png"> |

Here is a selection of images generated at 128x128

|128x Selection|
|:-----------:|
|<img src="https://raw.githubusercontent.com/DataSnaek/DCGAN-Keras/master/data/output/grid/out-64.png" width="80%"> |

### Space Images

|128x Selection|
|:-----------:|
|<img src="https://raw.githubusercontent.com/DataSnaek/DCGAN-Keras/master/data/output/grid/148650.jpg" width="120%"> |

# Getting Started

This section talks about how to use this model, its prerequisites and its paramaters.

## Prerequisites
This model was built using the following packages and versions (earlier versions may still work):

### DCGAN.py
```
- Python 3.6
- tensorflow/tensorflow_gpu 1.11
- Keras 2.2.4
- Pillow 5.1.0
- numpy 1.14.5
- scipy 1.1.0
- Ideally GPU/CUDA support setup with tensorflow_gpu, otherwise training will take a very long time
```
### scrape_imgs.py
```
- Python 3.6
- requests 2.18.4
- bs4 0.0.1
```
### resize_imgs.py
```
- Pillow 5.1.0
```
## Parameters for DCGAN.py

List of paramaters for the DCGAN.py file:

* ```--load_generator```: Path to existing generator weights file
  * e.g. ```../data/models/generat.h5```
* ```--load_discriminator```: Path to existing discriminator weights file
  * e.g. ```../data/models/discrim.h5```
* ```--data```: Path to directory of images of correct dimensions, using *.[filetype] (e.g. *.png) to reference images
  * e.g. ```../data/resized/paintings_256x/*.png```
* ```--sample```: If given, will generate that many samples from existing model instead of training
  * e.g. ```20```
* ```--sample_thresholds```: The values between which a generated image must score from the discriminator
  * e.g. ```(0.0,0.1)```
* ```--batch_size```: Number of images to train on at once
  * e.g. ```24```
* ```--image_size```: Size of images as tuple (height,width). Height and width must both be divisible by (2^5)
  * e.g. ```(192,256)```
* ```--epochs```: Number of epochs to train for
  * e.g. ```500000```
* ```--save_interval```: How many epochs to go between saves/outputs
  * e.g. ```100```
* ```--output_directory```: Directoy to save weights and images to
  * e.g. ```../data/output```

 ## Example Usage

 ### DCGAN.py
 To train a fresh model with default hyper parameters:
  ```
  python3 DCGAN.py
  ```

 To train a fresh model on some data, the following command template is ideal:

 ```
 python3 DCGAN.py --data /data/images/*.png --epochs 100000 --output /data/output
 ```

 Modifications can be made to image size, batch size etc. using the parameters. If your GPU doesn't have enough memory, you can change the size of the filters within the file, the image size and the batch size to better suit your GPU capability.

 ### scrape_imgs.py

 In it's current state it will download all of the images on [this ArtUK page]("https://artuk.org/discover/artworks/search/class_title:landscape--category:countryside/page/0"). You can modify the URL given in the file with any page or set of categories on ArtUK and it will download those instead.

 ### resize_imgs.py

 Will resize any directory of imgs to the specified size and store the new imgs in a different directory

 # Licensing

 This project is released under the MIT license, see LICENSE.md for more details.
