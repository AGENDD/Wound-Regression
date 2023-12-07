import numpy as np
from sklearn.metrics import mean_squared_error
import time
from tqdm import tqdm
# =============================================================================
# def main():
# =============================================================================
t1 = time.time()
X = np.load('X_SHP_ReZ_CvtL211.npy') # load preprocessed data
Y = np.load('Y.npy')

from sklearn.ensemble import RandomForestRegressor
RT = []
Ysum = np.zeros(Y.shape)

for i in tqdm(range(4)):            # 4片独立的随机森林
    RT.append(RandomForestRegressor(n_estimators=105,
                                    random_state=114514+i*1919,
                                    max_depth=11,
                                    min_samples_split=6,
                                    min_samples_leaf=9,
                                    max_features=0.1))
    RT[i].fit(X,Y - Ysum)           # 做差, 对前一片产生的误差
    Ysum += RT[i].predict(X)

MSE=mean_squared_error(Y,Ysum)
print('Training MSE=',MSE)

import pickle
pickle.dump(RT,open('boostRT.sav','wb'))

t2 = time.time()
print("time is ",t2-t1)



