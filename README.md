# Image Caption Joint Embedding

A multimodal embedding of images and captions, built with PyTorch, written with Python3.

## Introduction

Inspired by the paper: https://github.com/linxd5/VSE_Pytorch

## Getting started

Place your data in the `data/` folder. By default, the repository is setup to use the `f8k` dataset:
   * `wget http://www.cs.toronto.edu/~rkiros/datasets/f8k.zip`
   * To use your own data, you must format it like the f8k and change the dataset name in `settings.py`. 
      * It must follow the same naming convention as `f8k`.
   * You **must** also create an empty folder called `dict` in the project directory.
   * Your project directory should look like so:
  
    ![file](https://i.imgur.com/VvgsZIy.png)
    
## Train the model

Run `python3 train.py` to train your model. The best model will be saved automatically as `best.pkl`. Open `settings.py` to edit the configuration of the model and training.

## Test the model

Run `python3 test.py`. You must have a test dataset that follows the same input format and naming convention as the f8k otherwise you will get errors.

The test will output the averaged `Recall@K` scores.
