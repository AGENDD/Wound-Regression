from PIL import Image, ImageDraw
import time
import numpy as np
from sklearn.metrics import mean_squared_error
import csv
from tqdm import tqdm

def readImage(rootpath): 
    pics=[]     # only to read original pictures
    
    prefix = rootpath + '/' 
    gtFile = open(prefix + 'myData'+ '.csv')
    gtReader = csv.reader(gtFile, delimiter=';')
    next(gtReader)
    
    for row in tqdm(gtReader):
            pics.append(Image.open(prefix + row[0]))

    gtFile.close()
    return pics

def darwRectangle(imgs, Y, Ypred): # visualize the prediction outputs
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
t1 = time.time()
TestPics = readImage('Test')                # read test images

X = np.load('TestX_SHP_ReZ_CvtL211.npy')    # load prepocessed test data
Y = np.load('TestY.npy')

import pickle
RT=pickle.load(open('boostRT.sav', 'rb'))   # load model boostRTs

Ypred = 0
for i in range(len(RT)):
    Ypred += RT[i].predict(X)               # boostRTs gathering

MSE=mean_squared_error(Y,Ypred)
print('Test MSE=',MSE)

darwRectangle(TestPics, Y, Ypred)           # visualize the prediction

for la in (range(len(Ypred))):
    print(f'Output:{Ypred[la]},\t Label:{Y[la]}')


t2 = time.time()
print("time is ",t2-t1)





