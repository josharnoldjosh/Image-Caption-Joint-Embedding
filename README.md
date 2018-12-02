# Image Caption Joint Embedding

A multimodal embedding of images and captions, built with PyTorch, written with Python3.

## Introduction

Inspired by the paper: https://github.com/linxd5/VSE_Pytorch

## Getting started

Place your data in the `data/<dataset_name>/` folder. To find out more about preprocessing your data, please check out this repository: https://github.com/josharnoldjosh/Deep-Fashion-Joint-Embedding-Preprocessing The repository formats a popular dataset to train as a joint embedding. You can follow the code, it is fairly easy to understand.

In addition, make sure you have an empty folder called `dict`.
    
## Train the model

Run `python3 train.py` to train your model. Run `python3 test.py` to test your model.

Alot of work has been done behind the scenes. As long as your input data is formatted properly, everything should be good. The only three files you will need to worry about is `train.py`, `test.py`, and `settings.py`. 

The script has been setup to use k-fold cross validation. Make sure you configure `settings.py`. Everything else should be straight forward.