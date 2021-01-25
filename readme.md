# Neural network training, using CIFAR-10 dataset
A showcase project for my coding practices and use of deep learning.

## Dataset
The CIFAR-10 dataset consists of 60 000 32x32 colour images in 10 classes, with 6000 images per class. There are 50 000 training images and 10 000 test images.

Below are 10 images visualized from each class:
<div style=overflow:hidden;>
<img align="left" width="470" height="300" src="readme_images/cifar10_visualization.PNG">
</div>

## Results
With my trained neural network I was able to achieve xxxx% accuracy.

Below the image of resulting confusion matrix. Correct answers are shown on diagonal.

%Image of confusion matrix%

## Main features of this project
This project shows skills in following areas:

Pytorch, Docker, Git, file mangaement, commentting, hyperparamenter optimization, dataset management and transformation, knowledge in common neural networks and creation of neural networks and lastly correct model training, testing and early stoping practices.

## Hyperparameter optimization
Hyperparameter optimization was done using wandb and xxxx runs were done. I used bayesian optmization with hyperband early stops.

Hyperparameters used were:
- Base Network
- Learning rate
- Weight decay
- 7 on and off augments
- Augmentation rate

For optimizer I used Adam and for loss I used CrossEntropyLoss.

%Image of wandb parameters tested%

# Usage instructions
## Pre-requisites
There are two ways to set up needed librarys: Docker and requirements.txt

If you use gpu make sure you have nvidia graphics card and installed nesessary drivers and cuda. The docker container contains it's own cuda.

### a) Docker
If you are using windows and want to use gpu with container follow [these instructions.](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

Not tested, but if you want to run this with only cpu and windows you could install [docker desktop for windows.](https://docs.docker.com/docker-for-windows/install/)

For linux systems you can find nvidia-docker2 install instructions [here.](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)


You may need to start docker using command:
- sudo systemctl restart docker
- or 
- sudo service start docker

Building docker image:
cd to this_project base folder and use command: docker build --pull --rm "Docker" -t class:latest

Running docker image:
docker run -t -d --gpus all --mount type=bind,source=*PATH TO THIS PROJECTS BASE FOLDER*,target=/workspaces/CVproject class:latest

Then connect to the container with commands:
- docker ps
- Look at the runnin CONTAINER_ID 
- docker exec -it CONTAINER_ID bash
- cd /workspaces/cvproject

You are now able to execute commands and can move to Usage part of instructions.

### b) requirements.txt
Create new [conda enviroment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or activate an existing one. Note that you will need to take care of installing CUDA by yourself. Go to this_project/Docker folder and then use commands:
 - apt update && apt install python3-pip -y
 - apt-get install cython3
 - pip3 install -r requirements.txt
 - apt update && apt install -y libsm6 libxext6
 - apt-get install -y libxrender-dev

## Usage
Train a model with specified parameters or start an hyperparameter optimization. You will need internet connection to download datasets, models and use hyperparameter optimization.

### a) Run with specified parameters
To run with default parameters just command: python3 classifier.py

To find all parameters use python3 classifier.py --help

### b) Hyperparameter optimization
To use the hyperparameter optimization create an [wandb account.](https://wandb.ai/site)
On wandb site after registering create a new project.

Then on bash cd to this_project_folder and use commands:
- wandb on
- wandb sweep -project=*name_of_created_project* sweep.yml
- Use the wandb agent XXXXX/XXXX/XXXX command given by previous step to start the optimization.

If you want to change hyperparameters available change contents of sweep.yml. You can find all parameters using python3 classifier.py --help or go to options.py to look for parameters.