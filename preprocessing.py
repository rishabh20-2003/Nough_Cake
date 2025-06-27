# preprocessing.py

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import savgol_filter

class SNV(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        return (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, ddof=1, keepdims=True)

class BaselineCorrection(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        baseline = savgol_filter(X, window_length=11, polyorder=2, axis=1)
        return X - baseline

class SavitzkyGolay(BaseEstimator, TransformerMixin):
    def __init__(self, window_length=11, polyorder=2, deriv=1):
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
    def fit(self, X, y=None): return self
    def transform(self, X):
        wl = min(self.window_length, X.shape[1]) if X.shape[1] % 2 == 1 else X.shape[1] - 1
        return savgol_filter(X, window_length=wl, polyorder=self.polyorder, deriv=self.deriv, axis=1)

class MSC(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.mean_spectrum_ = X.mean(axis=0)
        return self
    def transform(self, X):
        X_msc = np.empty_like(X)
        for i in range(X.shape[0]):
            fit = np.polyfit(self.mean_spectrum_, X[i, :], deg=1)
            X_msc[i, :] = (X[i, :] - fit[1]) / fit[0]
        return X_msc
