import lightgbm as lgb
import xgboost as xgb
from skl2onnx.algebra.sklearn_ops import OnnxSklearnStackingClassifier
from sklearn.ensemble import StackingClassifier


class ChurnModel():

    """Customer churn classifier
    """

    def __init__(self):
        pass

    def churn_model(self, parameters) -> OnnxSklearnStackingClassifier:
        self.lgbm_paramaters = parameters.lgbm_params
        self.xgb_paramaters = parameters.xgb_params
        estimators = [
            ('lgbm', lgb.LGBMClassifier(**self.lgbm_paramaters,verbose=-1)),
            ('xg', xgb.XGBClassifier(enable_categorical=True,**self.xgb_paramaters)),
            ('lgbm2', lgb.LGBMClassifier(**self.lgbm_paramaters,verbose=-1))
            ]
        return  OnnxSklearnStackingClassifier(
                                            estimators=estimators,
                                            final_estimator=xgb.XGBClassifier(enable_categorical=True,**self.xgb_paramaters),
                                            passthrough=False
                                            )
