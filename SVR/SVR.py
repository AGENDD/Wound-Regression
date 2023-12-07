from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import numpy as np

# 假设 X 是你的图片特征，y 是你的目标值
X = np.random.rand(100, 1000)  # 100张图片，每张图片1000个特征
y = np.random.rand(100, 4)  # 100张图片，每张图片有4个目标值

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 对于每个目标值训练一个 SVR 模型
models = []
for i in range(4):
    model = SVR()
    model.fit(X_train, y_train[:, i])
    models.append(model)

# 预测
predictions = []
for model in models:
    prediction = model.predict(X_test)
    predictions.append(prediction)

predictions = np.array(predictions).T  # 转置以匹配目标值的形状
