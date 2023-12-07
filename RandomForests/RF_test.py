import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageDraw
import math
import csv
import numpy as np
from sklearn.metrics import mean_squared_error
import time
from tqdm import tqdm

def readImage(rootpath):
    pics=[]
    
    prefix = rootpath + '/' 
    gtFile = open(prefix + 'myData'+ '.csv')
    gtReader = csv.reader(gtFile, delimiter=';')
    next(gtReader)
    
    for row in tqdm(gtReader):
            pics.append(Image.open(prefix + row[0]))
    
    gtFile.close()
    return pics

def darwRectangle(imgs, Y, Ypred):
    for i in tqdm(range(len(imgs))):
        draw = ImageDraw.Draw(imgs[i])
        
        [x,y,xlen,ylen] = Y[i]
        rec = [(x-xlen//2,y-ylen//2), (x+xlen//2,y+ylen//2)]
        
        [x,y,xlen,ylen] = Ypred[i]
        pred = [(x-xlen//2,y-ylen//2), (x+xlen//2,y+ylen//2)]
        
        thickness = 5

        for j in range(thickness):
            rec_t= [(rec[0][0]-j,rec[0][1]-j),(rec[1][0]+j,rec[1][1]+j)]
            pred_t= [(pred[0][0]-j,pred[0][1]-j),(pred[1][0]+j,pred[1][1]+j)]
            
            draw.rectangle(rec_t, outline ="red")
            draw.rectangle(pred_t, outline ="blue")
        
        imgs[i].save('Test_out/pictue' + str(i) + '.jpg')
    return
# =============================================================================
# def main():
# =============================================================================
TestPics = readImage('Test')        # read test images
t1 = time.time()
X = np.load('TestX_SHP_ReZ_CvtL211.npy')    # load preprocessed data
Y = np.load('TestY.npy')

import pickle
reg=pickle.load(open('RT.sav', 'rb'))
Ypred=reg.predict(X)

MSE=mean_squared_error(Y,Ypred)
print('Training MSE=',MSE)

darwRectangle(TestPics, Y, Ypred)

t2 = time.time()
print("time is ",t2-t1)





