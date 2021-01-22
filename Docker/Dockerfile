FROM nvidia/cuda:11.0-cudnn8-runtime-ubuntu18.04

COPY requirements.txt .
RUN apt update && apt install python3-pip -y
RUN apt-get install cython3
RUN pip3 install -r requirements.txt
