import os
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor 
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.metrics import mean_squared_error
# =============================================================================
# def main():
# =============================================================================

# =============================================================================
# Ctr = ['squared_error','poisson','absolute_error','friedman_mse']
# Ctr2 = ['均方误差','泊松偏差','绝对平均误差','费尔曼均方误差']
# for i in tqdm(range(len(Ctr))):
#     RT = RandomForestRegressor(criterion=Ctr[i],random_state=114514)
#     score = cross_val_score(RT,X,Y,cv=5).mean()
#     print(f'{Ctr2[i]}得分: {score}')
# =============================================================================
    # To figure out what criterion to use by crossvalidation.
t1 = time.time()
sub_t1 = time.time()

X = np.load('X_SHP_ReZ_CvtL211.npy')
Y = np.load('Y.npy')
train =     []
train_out = []
test =      []
test_out =  []

def preSeperate(cv):
    sublen = len(X)/cv
    for i in range(cv):
        tr =     []
        tr_out = []
        te =      []
        te_out =  []
        l = sublen * i
        r = sublen * (i+1) - 1
        for j in range(len(X)):
            if l <= j and j <= r:
                te.append(X[j])
                te_out.append(Y[j])
            else:
                tr.append(X[j])
                tr_out.append(Y[j])
        train.append(tr)
        train_out.append(tr_out)
        test.append(te)
        test_out.append(te_out)
    return
        
param_ = []
param_.append(list(range(104,112))) # n_estimators
param_.append(list(range(10,15)))   # max_depth
param_.append(list(range(4,7)))     # min_samples_split
param_.append(list(range(9,15)))    # min_samples_leaf
param_.append([0.1,0.2])            # min_samples_leaf

P = np.zeros((len(param_)))
T = 0

MseAll = []
def DFSparameters(layer,deepestLayer):
    if layer > deepestLayer:
        mse = []
        for j in range(5):
            RT = []
            tmp_sum = np.zeros(np.array(train_out[0]).shape)
            for k in range(4):
                RT.append(RandomForestRegressor(n_estimators=int(P[0]),
                                                random_state=114514+k*1919,
                                                max_depth=int(P[1]),
                                                min_samples_split=int(P[2]),
                                                min_samples_leaf=int(P[3]),
                                                max_features=0.1))
                RT[k].fit(train[j],train_out[j] - tmp_sum)
                sub_pred = RT[k].predict(train[j])
                tmp_sum += sub_pred
                
            # test_pred = np.zeros(np.array(test_out[0]).shape)
            test_pred = 0.0
            for k in range(4):
                test_pred += RT[k].predict(test[j])
                
            mse.append(mean_squared_error(test_out[j],test_pred))
          
        global T,sub_t1
        T += 1
        MseAll.append([T,np.mean(mse)])
        print(f'time:{int(time.time()-sub_t1)},totally:{int(time.time()-t1)} {T}*')
        sub_t1 = time.time()
        
        # os.system('say please dont touch me! I m runing dee pi thon program.')
        # os.system(f'say 已完成第{T}片随机森林！')
    else:
        for i in range(len(param_[layer])):
            P[layer] = param_[layer][i]
            DFSparameters(layer+1,deepestLayer)
    
# =============================================================================
# def main():
# =============================================================================
preSeperate(5)      # 预先分割好5组{training+validating}, 并储存

DFSparameters(0,4)  # 网络状在给出的范围内搜索 最优参数搭配
    
MseAll = np.array(MseAll)

min_mse = np.where(MseAll==np.min(MseAll[:,1]))[0][0] 
                    # 找出最高得分对应的索引
print("最优方案序号 以及最低 MSE:",MseAll[min_mse])
plt.figure(figsize=[20,5])
plt.plot(MseAll[:,0],MseAll[:,1])
plt.show()

t2 = time.time()
print("time is \n",t2-t1)







