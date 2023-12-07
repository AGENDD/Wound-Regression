import os
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor 
import numpy as np
import time
from tqdm import tqdm

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
# =============================================================================
# def main():
# =============================================================================
t1 = time.time()
X = np.load('X_SHP_ReZ_CvtL211.npy')
Y = np.load('Y.npy')
preSeperate(5)

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

# kernel_ = 'rbf'
para_epsilon = [0.9,0.9,0.01,0.01]
para_C = [371.1, 514.54, 712.11, 672.72]
best = []

param_grid ={'C':[]}

MseAll = []
for i in tqdm(range(4)): # 固定‘epsilon’参数，寻找最优’C‘参数
    l = para_C[i]-20
    r = para_C[i]+20
    param_grid['C']=np.linspace(l,r,200)
    svr_reg = SVR(epsilon=para_epsilon[i])
    
    grid_search = GridSearchCV(svr_reg, param_grid, cv=5) # 交叉检验
    grid_search.fit(X,Y[:,i])
    
    best.append(grid_search.best_params_['C'])

print(best)
t2 = time.time()
print("time is ",t2-t1)


