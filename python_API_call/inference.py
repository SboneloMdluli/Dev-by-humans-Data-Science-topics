import os
import pprint
import sys

import numpy as np
import onnxruntime as rt
import pandas as pd
from sklearn.model_selection import train_test_split

parent_directory = os.path.abspath('..')
sys.path.append(parent_directory)

from model import params

df = pd.read_csv('../data/Train.csv')

df2 = df.copy()

X = df.drop(columns=['Target','ID','mrg','tenure','zone1','zone2','region','top_pack'])
y = df['Target']

X_train, X_test, y_train, y_test= train_test_split(X, y,stratify=y,test_size=0.20,random_state=params.SEED)

sess = rt.InferenceSession("../model_storage/boost_clf.onnx", providers=rt.get_available_providers())
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name


pred_onx = sess.run([label_name], {input_name: np.array(X_test).astype(np.float32)})[0]

pprint.pprint(np.array(X_test).astype(np.float32))

print("-------------------------------")

pprint.pprint(pred_onx)