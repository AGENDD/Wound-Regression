import os
import random
import cv2
import numpy as np
import csv
from tqdm import tqdm

def readImageData(rootpath):
    images = [] 
    output_1 = []
    
    prefix = rootpath + '/' 
    gtFile = open(prefix + 'myData'+ '.csv') 
    gtReader = csv.reader(gtFile, delimiter=';') 
    next(gtReader)
    for row in gtReader:
            images.append(prefix + row[0]) 
            output_1.append([int(row[1]),int(row[2]),int(row[3]),int(row[4])]) # the 8th column is the label
    gtFile.close()
    return images, output_1




def shift_image(image, dx, dy):
    rows, cols, _ = image.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(image, M, (cols, rows))
    return shifted



images, outputs = readImageData('Training')



count = 1

new_images = []
new_outputs = []
for j in tqdm(range(len(images))):

    image = cv2.imread(images[j])
    output = outputs[j]

    # 对每张图片做5次随机平移
    for i in range(5):
        dx, dy = random.randint(-100, 100), random.randint(-100, 100)  # 假设平移的范围是-50到50像素
        shifted = shift_image(image, dx, dy)

        # 保存平移后的图片
        cv2.imwrite(f'Training_Aug/image_{count}.jpg',shifted)
        new_images.append(f'image_{count}.jpg')
        thisoutput = output[:]
        thisoutput[0] += dx
        thisoutput[1] += dy
        new_outputs.append(thisoutput)
        count+=1

        
with open("Training_Aug/myData.csv","w") as f:
     f.write("filename;x;y;x_width;y_width\n")

     for i in range(len(new_images)):
          f.write(f"{new_images[i]};{new_outputs[i][0]};{new_outputs[i][1]};{new_outputs[i][2]};{new_outputs[i][3]}\n")

