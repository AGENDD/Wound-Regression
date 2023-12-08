import os
import numpy as np
from PIL import Image
import csv
import torch
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
import csv
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

# 计算图片3 channels mean 和 std 
def meanStd(folder_path):

    # 初始化变量
    mean = np.zeros(3)
    std = np.zeros(3)
    num_pixels = 0

    # 遍历文件夹中的所有图片
    for filename in os.listdir(folder_path):

        if filename.endswith(".JPG"):  # 可以根据你的图片格式进行修改
            # 加载图片
            img = Image.open(os.path.join(folder_path, filename))
            img = np.array(img) / 255.0  # 将像素值转换到[0,1]区间

            # 更新平均值和标准差
            mean += img.mean(axis=(0, 1)) * img.shape[0] * img.shape[1]
            std += img.std(axis=(0, 1)) * img.shape[0] * img.shape[1]
            num_pixels += img.shape[0] * img.shape[1]

    # 计算总的平均值和标准差
    mean /= num_pixels
    std /= num_pixels

    print("Mean: ", mean)
    print("Std: ", std)
# 计算output mean 和 std 
def meanStd2(folder_path):
    
    # 初始化变量
    mean = np.zeros(4)
    std = np.zeros(4)
    num_pixels = 0

    prefix = folder_path + '/' 
    gtFile = open(prefix + 'myData'+ '.csv') 
    gtReader = csv.reader(gtFile, delimiter=';') 
    next(gtReader)

    data = [[],[],[],[]]
    for row in gtReader:
        data[0].append(float(row[1]))
        data[1].append(float(row[2]))
        data[2].append(float(row[3]))
        data[3].append(float(row[4]))

    data[0] = np.array(data[0])
    data[1] = np.array(data[1])
    data[2] = np.array(data[2])
    data[3] = np.array(data[3])

    for i in range(4):
        mean[i] = np.mean(data[i])
        std[i] = np.std(data[i])

    print("Mean: ", mean)
    print("Std: ", std)
# 设置随机数种子
def seed(seed):
    # 设置种子
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# datalaoder 所需的数据集类
class MyDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = torch.tensor(labels)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
# 读取图片数据
def readImageData(rootpath,mm,ss):
    images = [] 
    output_1 = []
    
    prefix = rootpath + '/' 
    gtFile = open(prefix + 'myData'+ '.csv') 
    gtReader = csv.reader(gtFile, delimiter=';') 
    next(gtReader)
    for row in gtReader: 
            images.append(prefix + row[0]) 
            output_1.append([(float(row[1])- MeanOut[0])/StdOut[0] ,(float(row[2]) - MeanOut[1])/StdOut[1],(float(row[3]) - MeanOut[2])/StdOut[2],(float(row[4]) - MeanOut[3])/StdOut[3]]) # the 8th column is the label
    gtFile.close()
    return images, output_1

# 设置种子
seed(114514)

# 调用CUDA
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")


### 读取数据
print("Reading Data...")
trainPath = 'Training_Aug'
testPath = 'Test'


# meanStd2(trainPath)
# meanStd(trainPath)

Mean =  [0.63969549,0.60957314,0.60424098]
Std =   [0.22423368,0.23694334,0.24343221]

MeanOut =  [1607.54666667,1179.92,617.59333333,606.41333333]
StdOut = [29.81064389,37.59424069,69.9493242,68.13018779]

Data, Outputs = readImageData(trainPath,MeanOut,StdOut)
# 数据集分割
images_train, images_val, output_1_train, output_1_val = train_test_split(Data, Outputs, test_size=0.2, random_state=42)





### 预处理
print("Preprocessing...")
transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=Mean, std=Std),
    ])
# 建立数据集对象
train_dataset = MyDataset(images_train, output_1_train, transform)
val_dataset = MyDataset(images_val, output_1_val, transform)

# 建立dataloader
batch_size = 64 
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


### 建立模型
print("Building Model...")
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 4)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)



### 训练
print("Training...")
epoch_num = 300

train_loss = []
val_loss = []

model = model.to(device)
for epoch in tqdm(range(epoch_num)):  # loop over the dataset multiple times

    #训练周期
    model.train()
    trainloss = 0.0
    trainTime = 0
    for inputs, labels in tqdm(train_dataloader,leave=False):

        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward + backward + optimize

        # zero the parameter gradients
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)



        loss.backward()
        optimizer.step()

        trainloss += loss.item()*inputs.size(0)
        trainTime += inputs.size(0)
    # print(f"Epoch {epoch + 1}, loss: {trainloss / trainTime}")

    # 验证周期
    model.eval()  # 设置模型为评估模式
    valloss = 0.0
    valTime = 0
    with torch.no_grad():  # 在验证阶段不需要计算梯度
        for inputs, labels in tqdm(val_dataloader,leave=False):

            inputs = inputs.to(device)
            labels = labels.to(device)


            outputs = model(inputs)
            loss = criterion(outputs, labels)
            valloss += loss.item()*inputs.size(0)
            valTime += inputs.size(0)

    # print(f"Epoch {epoch + 1}, validation loss: {valloss / valTime}")

    # train_loss.append(trainloss/trainTime)
    # val_loss.append(valloss/valTime)

print("Finished Training")
     

### 保存数据
print("Saving...")
torch.save(model,"neuralNetwork_new.pth")
# count = 1
# with open("neuralNetwork_aug.csv",'w') as f:
#     # print("epoch,trainLoss,valLoss\n")
#     for count in range(len(train_loss)):
#         f.write(f"{count+1},{train_loss[count-1]},{val_loss[count-1]}\n")

