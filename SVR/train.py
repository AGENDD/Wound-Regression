import numpy as np
from sklearn.metrics import mean_squared_error
import time
from tqdm import tqdm
# =============================================================================
# def main():
# =============================================================================
t1 = time.time()
X = np.load('X_SHP_ReZ_CvtL211.npy')

Y = np.load('Y.npy')

# para_C=[421.163,512.504,708.97,673.0] # 'C', old
para_C=[364.95, 506.80, 713.22, 673.42] # 'C'
para_e=[0.9, 0.9, 0.01, 0.01]           # 'epsilon'
# parameters found by crossvalidation

from sklearn.svm import SVR
svr_reg = []
Ypred = np.zeros(Y.shape)
for i in tqdm(range(4)):
    svr_reg.append(SVR(C=para_C[i],epsilon=para_e[i]))
    
    svr_reg[i].fit(X,Y[:,i])
    Ypred[:,i]=svr_reg[i].predict(X)

MSE=mean_squared_error(Y,Ypred)
print('Training MSE=',MSE)

t2 = time.time()
print("time is ",t2-t1)

import pickle
pickle.dump(svr_reg,open('SVR.sav','wb'))


