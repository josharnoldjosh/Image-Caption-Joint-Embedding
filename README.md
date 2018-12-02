# Image Caption Joint Embedding

A multimodal embedding of images and captions, built with PyTorch, written with Python3.

## Introduction

Inspired by the paper: https://github.com/linxd5/VSE_Pytorch

## Getting started

Place your data in the `data/` folder. The input format is essentially a file containing captions separated by a new line, and a numpy array that contains the image features. The caption and image features are related by index, e.g, `captions[0]` and `images[0]` would represent a pair.

To find out more about preprocessing your data, please check out this repository: https://github.com/josharnoldjosh/Deep-Fashion-Joint-Embedding-Preprocessing

The repository formats a popular dataset to train as a joint embedding. You can follow the code, it is fairly easy to understand.

Make sure you have an empty folder called `dict`.
  
![file](https://i.imgur.com/VvgsZIy.png)
    
## Train the model

Run `python3 train.py` to train your model. Run `python3 test.py` to test your model.

Alot of work has been done behind the scenes. As long as your data is formatted in the right way for inputting into the folder, everything should be good. The only three files you will need to worry about is `train.py`, `test.py`, and `settings.py`. 

The script has been setup to use k-fold cross validation. Make sure you configure `settings.py`. Everything else should be straight forward.