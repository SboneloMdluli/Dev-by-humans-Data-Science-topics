import os
import pickle
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from onnxmltools.convert import convert_lightgbm
from skl2onnx.common.data_types import FloatTensorType
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

parent_directory = os.path.abspath('..')
sys.path.append(parent_directory)

from data_transformation.transform import CategoricalTransform, NumericalTransform
from model import churn_model, params
from utils.common import Common

model = churn_model.ChurnModel()

df = pd.read_csv('../data/Train.csv')
df2 = df.copy()

X = df.drop(columns=['Target','ID','mrg','tenure','zone1','zone2'])
y = df['Target']
X_train, X_test, y_train, y_test= train_test_split(X, y,stratify=y,test_size=0.20,random_state=params.SEED)
categorical_features = X.select_dtypes(exclude='number').columns.tolist()
numerical_features = X.select_dtypes(include=np.number).columns.to_list()

# define pipeline for reusability
pipeline = Pipeline(
    steps=[
        ("numerical transformation", NumericalTransform(columns=numerical_features)),
        ("categorical transformation", CategoricalTransform(columns=categorical_features)),
        ("train model", model.churn_model(params))
        ],
)

common = Common()

pred: Pipeline = pipeline.fit(X_train,y_train)

with open('../model_storage/stacked.pkl', 'wb') as f:
    pickle.dump(pred, f)

binary_predictions,optimal_threshold = common.threshold_calibration(pred,X_test,y_test)
report = classification_report(y_test, binary_predictions, target_names=params.TARGET_NAMES)
print(report)
print(optimal_threshold)
f1_thres=f1_score(y_test, binary_predictions)
print(f1_thres)


""" Onnx conversion"""

X: pd.DataFrame = X.drop(columns=categorical_features)
y = df2['Target']
X_train, X_test, y_train, y_test= train_test_split(X, y,stratify=y,test_size=0.20,random_state=params.SEED)

model_lgb = lgb.LGBMClassifier(**params.lgbm_params,verbose=-1)
model_lgb.fit(X_train,y_train)

#convert and save model in onnx format
initial_type = [('float_input', FloatTensorType([None, 11]))]
onx = convert_lightgbm(model_lgb, initial_types=initial_type, target_opset=9,zipmap=False)
with open("../model_storage/boost_clf.onnx", "wb") as f:
    f.write(onx.SerializeToString())
