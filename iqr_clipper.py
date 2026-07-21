import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class IQRClipper(BaseEstimator, TransformerMixin):

    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        self.Q1_ = np.percentile(X, 25, axis=0)
        self.Q3_ = np.percentile(X, 75, axis=0)

        self.IQR_ = self.Q3_ - self.Q1_

        self.lower_bound_ = self.Q1_ - self.factor * self.IQR_
        self.upper_bound_ = self.Q3_ + self.factor * self.IQR_

        return self

    def transform(self, X):
        return np.clip(
            X,
            self.lower_bound_,
            self.upper_bound_
        )