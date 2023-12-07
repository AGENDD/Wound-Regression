import matplotlib.pyplot as plt
import csv
from PIL import Image,ImageDraw
import numpy as np
from sklearn.metrics import mean_squared_error

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

#int main():
trainImagess, trainOutputs = readImageData('Training')

trainImages = trainImagess[:]
for i in range(len(trainImages)):
    trainImages[i]=trainImages[i].resize((80,60), Image.BICUBIC)
    trainImages[i]=trainImages[i].convert('L')
    trainImages[i]=np.array(trainImages[i])



# print number of historical images
print('number of historical data=', len(trainOutputs))
# show one sample image
# plt.imshow(trainImages[4])
# plt.show()

# design the input and output for model
X=[]
Y=[]
for i in range(0,len(trainOutputs)):
    # input X just the flattern image, you can design other features to represent a image
    X.append(trainImages[i].flatten())
    Y.append(trainOutputs[i])
X=np.array(X)
Y=np.array(Y)


#train a Randomforest
print("training")
from sklearn.ensemble import RandomForestRegressor
reg=RandomForestRegressor(n_estimators=100)
reg.fit(X,Y)
Ypred=reg.predict(X)

#check the accuracy
MSE=mean_squared_error(Y,Ypred)
print('Training MSE=',MSE)



# save model
import pickle
pickle.dump(reg,open('model.sav','wb'))

# with open('randomForest_Train.csv','w') as f:
#      for i in range(len(Y)):
#           f.write(f'{Y[i]},{Ypred[i]}\n')

                      
# print("Drawing")
# for i in range(len(Y)):

#     draw = ImageDraw.Draw(trainImagess[i])
#     top_left = (Y[i][0]-Y[i][2]/2,Y[i][1]-Y[i][3]/2)
#     bottom_right = (Y[i][0]+Y[i][2]/2,Y[i][1]+Y[i][3]/2)


#     draw.rectangle([top_left, bottom_right], outline='red',width=5)

#     top_left = (Ypred[i][0]-Ypred[i][2]/2,Ypred[i][1]-Ypred[i][3]/2)
#     bottom_right = (Ypred[i][0]+Ypred[i][2]/2,Ypred[i][1]+Ypred[i][3]/2)
#     draw.rectangle([top_left, bottom_right], outline='blue',width=5)

#     trainImagess[i].save(f'Train_out/pictue{i}.jpg')