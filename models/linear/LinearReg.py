from typing import Union, List, Optional

import numpy as np
import pandas as pd

from ..base.regressor import BaseRegressor


class LinReg(BaseRegressor):

    def __init__(self,
                 solver: List[str],
                 bias: bool =True,
                 lr: float = 0.001,
                 iters: int = 1000,
                 stop_iter: float = 1e-4):

        self.solver = solver
        self.bias = bias
        self.lr = lr
        self.iters = iters
        self.stop_iter = stop_iter
        self.w: Optional[np.ndarray] = None
        self.loss_history: List[float] = []


    def _add_bias(self, X: np.ndarray) -> np.ndarray:

        bias = np.ones(X.shape[0])

        return np.concatenate((bias, X), axis=1)


    def _preproc_data(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        pass


    def _fit_close(self, X: np.ndarray, y: np.ndarray) -> None:

        try:
            self.w = np.linalg.inv(X.T @ X) @ X.T @ y
        except np.linalg.LinAlgError:
            self.w = np.linalg.pinv(X.T @ X) @ X.T @ y


    def _fit_gradient(self, X: np.ndarray, y: np.ndarray) -> None:

        self.w = np.zeros(X.shape[1])
        prev = float('inf')

        for _ in range(self.iters):
            y_pred = X @ self.w
            error = y_pred - y
            loss = np.mean(error ** 2)

            self.loss_history.append(loss)

            if (prev - y) < self.stop_iter:
                break

            grad = (X.T @ error) / X.shape[0]
            self.w -= self.lr * grad

            prev = loss


    def fit(self, X: Union[pd.DataFrame, np.ndarray],
                  y : Union[pd.Series, np.ndarray] = None) -> np.ndarray:



        return self.w


    def transform(self):
        pass


    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        pass


    def get_features_imp(self) -> pd.Series:
        pass