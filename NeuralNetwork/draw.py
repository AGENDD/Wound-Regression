import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取CSV文件
data = pd.read_csv('neuralNetwork_Aug.csv', header=None)


plt.plot(data[0],data[1],color='red',label ="Train loss")
plt.plot(data[0],data[2],color='blue',label="Val loss")

max1_index = np.argmin(data[1])

max1_x = data[0][max1_index]
max1_y = data[1][max1_index]

max2_index = np.argmin(data[2])
print(max2_index)
max2_x = data[0][max2_index]
max2_y = data[2][max2_index]

# 添加水平和垂直的虚线
plt.axhline(y=max1_y, color='r', linestyle='--', linewidth=0.5)
plt.axvline(x=max1_x, color='r', linestyle='--', linewidth=0.5)

plt.axhline(y=max2_y, color='blue', linestyle='--', linewidth=0.5)
plt.axvline(x=max2_x, color='blue', linestyle='--', linewidth=0.5)

xticks = list(plt.gca().get_xticks())
yticks = list(plt.gca().get_yticks())

if max1_x not in xticks:
    xticks.append(max1_x)
if max1_y not in yticks:
    yticks.append(max1_y)
if max2_x not in xticks:
    xticks.append(max2_x)
if max2_y not in yticks:
    yticks.append(max2_y)

plt.gca().set_xticks(xticks)
plt.gca().set_yticks(yticks)

plt.xlim(0,301)
plt.ylim(0.0,5.0)


plt.savefig('loss_value_result_Aug.jpg')
plt.show()