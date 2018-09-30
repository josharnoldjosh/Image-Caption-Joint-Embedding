# Image Caption Joint Embedding

A multimodal embedding of images and captions, built with PyTorch, written with Python3.

## Introduction

Inspired by the paper: https://github.com/linxd5/VSE_Pytorch

## Getting started

Place your data in the `data/` folder. By default, the repository is setup to use the `f8k` dataset:
   * `wget http://www.cs.toronto.edu/~rkiros/datasets/f8k.zip`
   * To use your own data, you must format it like the f8k and change the import path in `data.py`
   * Your project directory should look like so:
   
   [![Screen_Shot_2018-09-30_at_12.52.17_AM.png](https://i.postimg.cc/XYZsWT2X/Screen_Shot_2018-09-30_at_12.52.17_AM.png)](https://postimg.cc/CRgGH6pV)

## Train the model

Run `python3 train.py` to train your model. The best model will be saved automatically as `best.pkl`. Open `settings.py` to edit the configuration of the model and training.
