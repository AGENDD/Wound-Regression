import os
import numpy as np
from PIL import Image,ImageDraw
import csv
import torch
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
import csv


Mean =  [0.63969549,0.60957314,0.60424098]
Std =   [0.22423368,0.23694334,0.24343221]

MeanOut =  [1607.54666667,1179.92,617.59333333,606.41333333]
StdOut = [29.81064389,37.59424069,69.9493242,68.13018779]


def readImageData(rootpath):
    '''Reads data 
    Arguments: path to the image, for example './Training'
    Returns:   list of images, list of corresponding outputs'''
    images = [] # images
    output_1 = [] # corresponding x index
    # there are other outputs like y index, x_width, y_width
    
    prefix = rootpath + '/' 
    gtFile = open(prefix + 'myData'+ '.csv') # annotations file
    gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
    next(gtReader)
    # loop over all images in current annotations file
    for row in gtReader:
            img=Image.open(prefix + row[0])  # the 1th column is the filename
            # preprocesing image, here we resize the image into a smaller one
            images.append(img) 
            output_1.append([float(row[1]),float(row[2]),float(row[3]),float(row[4])]) # the 8th column is the label
    
    gtFile.close()
    return images, output_1



print("loading model")
# 加载模型
import pickle
loaded_model = pickle.load(open('model.sav', 'rb'))


print("testing")







for i in range(len(tured)):
     for j in range(4):
          tured[i][j] = tured[i][j]*StdOut[j] + MeanOut[j]

for i in range(len(predicted)):
     for j in range(4):
          predicted[i][j] = predicted[i][j]*StdOut[j] + MeanOut[j]



for i in range(len(Test_origin)):

    draw = ImageDraw.Draw(Test_origin[i])
    top_left = (tured[i][0]-tured[i][2]/2,tured[i][1]-tured[i][3]/2)
    bottom_right = (tured[i][0]+tured[i][2]/2,tured[i][1]+tured[i][3]/2)


    draw.rectangle([top_left, bottom_right], outline='red',width=5)

    top_left = (predicted[i][0]-predicted[i][2]/2,predicted[i][1]-predicted[i][3]/2)
    bottom_right = (predicted[i][0]+predicted[i][2]/2,predicted[i][1]+predicted[i][3]/2)
    draw.rectangle([top_left, bottom_right], outline='blue',width=5)

    Test_origin[i].save(f'Test_out/pictue{i}.jpg')





