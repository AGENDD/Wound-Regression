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
    images = [] 
    output_1 = []

    prefix = rootpath + '/' 
    gtFile = open(prefix + 'myData'+ '.csv') 
    gtReader = csv.reader(gtFile, delimiter=';') 
    next(gtReader)
    for row in gtReader:
            img=Image.open(prefix + row[0])  
            images.append(img) 
            output_1.append([(float(row[1])- MeanOut[0])/StdOut[0] ,(float(row[2]) - MeanOut[1])/StdOut[1],(float(row[3]) - MeanOut[2])/StdOut[2],(float(row[4]) - MeanOut[3])/StdOut[3]]) # the 8th column is the label
    

    gtFile.close()
    return images, output_1



transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=Mean, std=Std),
    ])

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")


# 读取数据
testPath = 'Test'

Test_origin, TOutputs = readImageData(testPath)
Test = [transform(image).unsqueeze(0) for image in Test_origin]
Test = torch.cat(Test)
TOutputs = torch.tensor(TOutputs)



# 调用模型
print("loading model")
model = torch.load('neuralNetwork_aug.pth')
model.eval() 

print("testing")



criterion = torch.nn.MSELoss()
predicted = []
tured = []

total_loss = 0.0
with torch.no_grad():
    for i in range(len(Test)):

        inputs, labels = Test[i].unsqueeze(0), TOutputs[i]
        inputs = inputs.to(device)
        labels = labels.to(device)


        outputs = model(inputs)

        outputs = outputs.to(torch.device("cpu"))
        labels = labels.to(torch.device("cpu"))
        total_loss += criterion(outputs, labels).item()

        predicted.append((outputs).numpy()[0])
        tured.append(labels)
        print(f'Output:{(outputs)[0]}\tLabel:{labels}')

total_loss = total_loss/ len(Test)

print(f"normalized test loss:{total_loss}")
# test loss:0.7065473893657327  1673.5302734375
# aug test loss: 0.5394057413563133  1095.857421875



# 输出归一之前的结果

for i in range(len(tured)):
     tured[i] = tured[i].numpy()
     for j in range(4):
          tured[i][j] = tured[i][j]*StdOut[j] + MeanOut[j]

for i in range(len(predicted)):
     for j in range(4):
          predicted[i][j] = predicted[i][j]*StdOut[j] + MeanOut[j]

# print(tured)
# print(predicted)
tured = np.array(tured)
predicted = np.array(predicted)
# print(tured.shape)
# print(predicted.shape)
from sklearn.metrics import mean_squared_error

MSE = mean_squared_error(tured,predicted)
print(f"test loss:{MSE}")



# 打印图片结果

# for i in range(len(Test_origin)):

#     draw = ImageDraw.Draw(Test_origin[i])
#     top_left = (tured[i][0]-tured[i][2]/2,tured[i][1]-tured[i][3]/2)
#     bottom_right = (tured[i][0]+tured[i][2]/2,tured[i][1]+tured[i][3]/2)


#     draw.rectangle([top_left, bottom_right], outline='red',width=5)

#     top_left = (predicted[i][0]-predicted[i][2]/2,predicted[i][1]-predicted[i][3]/2)
#     bottom_right = (predicted[i][0]+predicted[i][2]/2,predicted[i][1]+predicted[i][3]/2)
#     draw.rectangle([top_left, bottom_right], outline='blue',width=5)

#     Test_origin[i].save(f'Test_out_aug/pictue{i}.jpg')





