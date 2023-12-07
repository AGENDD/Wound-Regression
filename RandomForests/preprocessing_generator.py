import csv
from PIL import Image, ImageFilter
import numpy as np
import math
from tqdm import tqdm

def getMeanStd(data, up): # up = 255 for images
    data/=up
    
    Mean = np.sum(data) / data.size
    Std = np.sum((data-Mean)**2.0)
        
    return Mean, math.sqrt(Std / data.size)

def normalization(data,up):
    [Mean, Std]=getMeanStd(data,up)
    
    return up, Mean, Std, (data-Mean)/Std

def ConvertL(data):
    tmp = np.zeros((len(data),len(data[0])))
    
    for i in range(len(data)):
        for j in range(len(data[0])):
            W=[2.0,1.0,1.0]                 # The weights of R,G,B
            tmp[i][j] = np.sum(data[i][j] * W) / sum(W)

    return tmp

def preprocessing(img):
    img=img.filter(ImageFilter.SHARPEN)     # 1st step: sharpen
    img=img.resize((80,60), Image.BICUBIC)  # 2nd step: resize
    img=np.array(img)
    img=ConvertL(img)                       # 3rd step: convert
    return img

def readImageData(rootpath):                # read
    pics=[]
    images=[]
    output_=[]
    
    prefix = rootpath + '/' 
    gtFile = open(prefix + 'myData'+ '.csv')
    gtReader = csv.reader(gtFile, delimiter=';')
    next(gtReader)
    
    for row in tqdm(gtReader):
            img=Image.open(prefix + row[0])
            pics.append(img)
            img=preprocessing(img)
            
            images.append(img) 
            output_.append([float(row[1]),float(row[2]),float(row[3]),float(row[4])])    
    
    gtFile.close()
    return pics, images, output_
# =============================================================================
# def main():
# =============================================================================
Original, trainImages, trainOutputs = readImageData('Training')
# Original, trainImages, trainOutputs = readImageData('Test')

X=[]
Y=[]
for i in range(len(trainOutputs)):
    X.append(trainImages[i].flatten())
    Y.append(trainOutputs[i])
X=np.array(X)
Y=np.array(Y)

[Xmax,Xmean,Xstd,X] = normalization(X,np.max(X))
# [Ymax,Ymean,Ystd,Y] = normalization(Y,np.max(Y))
# no need to normalize outputs Y

np.save('X_SHP_ReZ_CvtL211.npy', X)         # store the preproessed data
np.save('Y.npy', Y)
# np.save('TestX_SHP_ReZ_CvtL211.npy', X)
# np.save('TestY.npy', Y)

# loaded_data = np.load('output.npy')