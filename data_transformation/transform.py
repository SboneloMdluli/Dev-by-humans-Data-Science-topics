import pandas as pd
from skl2onnx.algebra.onnx_operator_mixin import OnnxOperatorMixin
from skl2onnx.algebra.onnx_ops import (
    OnnxConcatFromSequence,
    OnnxStringNormalizer,
    OnnxStringSplit,
)
from sklearn.base import BaseEstimator, TransformerMixin


class CategoricalTransform(BaseEstimator, TransformerMixin, OnnxOperatorMixin):

    """Perform predefined categorical transformations
    """

    def __init__(self,columns):
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)
        self.op_version = 20
        self.columns = set(columns)

    def fit(self, X:pd.DataFrame,y=None):
        return self

    def clean_string_column(self,df:pd.DataFrame, column_name:str) -> pd.DataFrame:
        df[column_name] = df[column_name].apply(lambda x: x.replace(' ', ''))
        # some instances have On-net other have on net
        df[column_name] = df[column_name].apply(lambda x: x.replace('-', ''))
        df[column_name] = df[column_name].apply(lambda x: x.replace(';', ','))
        df[column_name] = df[column_name].str.lower()
        return df

    def engineer_features(self, df :pd.DataFrame) -> pd.DataFrame:
        # concatenation of the region and top_pack feature
        df["top_pack_region_concat"] = df['region'].astype(str)+df['top_pack'].astype(str)
        # format features
        df['top_pack'] = df['top_pack'].astype('category')
        df['top_pack_region_concat'] = df['top_pack_region_concat'].astype('category')
        return df

    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        X[X.columns] = X[X.columns].astype('category')
        cleaned= self.clean_string_column(X,'top_pack')
        df = self.engineer_features(cleaned)
        return df


    def onnx_shape_calculator(self):
        def shape_calculator(operator):
            operator.outputs[0].type = operator.inputs[0].type
        return shape_calculator


    # We need to use some if else statments
    def to_onnx_operator(self, inputs=None, outputs=('Y', )):
        if inputs is None:
            raise RuntimeError("Parameter inputs should contain at least "
                               "one name.")
        i0 = self.get_inputs(inputs, 0)

        stopwords: list[str] = [" ",",",";"]

        return  OnnxStringNormalizer(
                            OnnxConcatFromSequence(
                                    OnnxStringSplit(
                                        i0, delimiter="",
                                        op_version=self.op_version
                                        ),
                                        op_version=self.op_version
                            ),
                            stopwords =stopwords,
                            op_version=self.op_version,
                            case_change_action="UPPER",
                            output_names=outputs
                )


class NumericalTransform(BaseEstimator, TransformerMixin, OnnxOperatorMixin):

    """Perform predefined categorical transformations
    """

    def __init__(self,columns):
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)
        self.columns = columns
        self.op_version = 12

    def fit(self, X:pd.DataFrame,y=None):
        return self


    def engineer_features(self, df :pd.DataFrame) -> pd.DataFrame:

        # average revenue spent per visit by a customer
        df["revenue_per_visit"] = 3*df['arpu_segment']/df['regularity']
        # a limited view of revenue per visit (looking at 90 day window)
        df["freq_regularity"] = df['frequency']*df['regularity']
        # calls in network belonging to by product1 (feature product for correctness )
        df["on_net_in_procuct"] = df['on_net']*df['Procuct_1']
        # ratio of revenue generated by top_pack (inverse for correctness)
        df["frequency_balance"] = df['freq_top_pack']/df['frequency']
        return df

    def transform(self, X:pd.DataFrame) -> pd.DataFrame:

        df = self.engineer_features(X)
        return df

    def onnx_shape_calculator(self):
        def shape_calculator(operator):
            operator.outputs[0].type = operator.inputs[0].type
        return shape_calculator
