# DeepSpeech2
Python3 installation and complete setup only for prediction of Baidu's deep speech 2 model.

Implementation of DeepSpeech2 architecture for ASR. It is an open-source
implementation of end-to-end Automatic Speech Recognition (ASR) engine, based on
Baidu&#39;s Deep Speech 2 paper, with PaddlePaddle platform. Biadu’s pre-trained model for
English is used for inference.

# System Requirements:
OS: Ubuntu 16.04.5 LTS
Language: Python3 and Bash
Database: None

# Tools:
PaddlePaaddle
SoundFile
Efficient Signal Resampling
Python Speech Feature extraction

# Packages:
List of Python packages has been attached in requirements.txt

# Environment setup

## Folders setup
sh folder_setup.sh


## Utilities setup

Either run required_packages.sh 

OR

sudo apt-get install python3-pip
sudo python3 -m pip install paddlepaddle

## Libraries setup

sudo python3 -m pip install -r requirements.txt

OR

sudo python3 -m pip install scipy==1.2.1
sudo python3 -m pip install SoundFile==0.9.0.post1
sudo python3 -m pip install resampy==0.1.5
sudo python3 -m pip install python_speech_features

## Other steps

Replace </data_utils/utility.py> from repo
