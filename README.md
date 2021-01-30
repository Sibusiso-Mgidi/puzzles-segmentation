# Puzzle Pieces Segmentation using U-Net
This project was done in partial fulfillment of the requirement for the Computer Vision course lectured by Dr. Richard Klein from the Witwatersrand University.

## About the Puzzle Pieces Segmentation project
The aim of this project is to segment puzzle pieces from an image by training the U-Net model to predict a segmentation mask on the puzzle images. This is a computer vision technique called Semantic Segmentation which classifies each pixel in an image to a predefined class. For this project, we have two classes namely the puzzle and background.
The U-Net implementation was done in Python, Tensorflow and Keras. The aim set out above was achieved bythe following objectives:
* Implementation of the U-Net
* We trained the U-Net model with pre-trained weights from VGG16.
* Applied 6-fold cross validation

 For this work, we answered the following questions:
* Does data augmentation improve U-Net with a limited dataset?
* Does k-fold cross validation increase models accuracy?

## Sample Dataset
To train the model, we used the puzzle dataset from the computer vision course. The dataset consists of 48 pre-processed puzzle pieces with corresponding masks having dimensions 768 x 1024 x 3 and shape 768 x 1024 respectively.
<p float="left">
  <img src="datasets/puzzle_corners_1024x768/masks-1024x768/image-36.png" width="425" />
</p>


