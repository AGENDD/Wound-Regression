# import matplotlib.pyplot as plt
# import csv
# from PIL import Image, ImageFilter
# import math
# from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error
import time
# =============================================================================
# def main():
# =============================================================================
t1 = time.time()
X = np.load('X_SHP_ReZ_CvtL211.npy')    # load preprocessed data
Y = np.load('Y.npy')

from sklearn.ensemble import RandomForestRegressor
reg=RandomForestRegressor(n_estimators=100,random_state=114514)
reg.fit(X,Y)
Ypred=reg.predict(X)

MSE=mean_squared_error(Y,Ypred)
print('Training MSE=',MSE)

t2 = time.time()
print("time is ",t2-t1)

import pickle
pickle.dump(reg,open('RT.sav','wb'))



