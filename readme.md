# Neural network training, using CIFAR-10 dataset
A showcase project for my coding practices and use of deep learning.

## Dataset
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

Below are 10 images visualized from each class:
![CIFAR10 visualization](readme_images/cifar10_visualization.PNG)

## Results
With my trained neural network I was able to achieve xxxx% accuracy.

Below the image of resulting confusion matrix. Correct answers are shown on diagonal.
%Image of confusion matrix%

## Hyperparameter optimization
Hyperparameter optimization was done using wandb and xxxx runs were done. I used bayesian optmization with hyperband early stops.

Hyperparameters used were:

    - Base Network
    - Learning rate
    - Weight decay

For optimizer I used Adam and for loss I used CrossEntropyLoss.

%Image of wandb parameters tested%

## Clear code and containers
The code writen is well commented and stuctured. The reposity is also well structured. There docker can be used to create the same enviorment.

%Image about comments and folder structure%

# Usage instructions
## Pre-requisites
There are two ways to set up needed librarys: Docker and requirements.txt
### a) Docker
### b) requirements.txt

## Usage
Train a model with specified parameters or start an hyperparameter optimization.

### a) Run with specified parameters
### b) Hyperparameter optimization
