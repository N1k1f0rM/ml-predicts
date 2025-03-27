from typing import Union, List, Optional, Literal
import numpy as np
import pandas as pd
from core.base_regressor import BaseRegressor


class LinReg(BaseRegressor):

    def __init__(self,
                 solver: Literal["close", "gd"] = "gd",
                 bias: bool = True,
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

        return np.c_[bias, X]

    def _preproc_data(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self._add_bias(X)

    def _fit_close(self, X: np.ndarray, y: np.ndarray) -> None:

        try:
            self.w = np.linalg.inv(X.T @ X) @ X.T @ y
        except np.linalg.LinAlgError:
            self.w = np.linalg.pinv(X.T @ X) @ X.T @ y

    def _fit_gradient(self, X: np.ndarray, y: np.ndarray) -> None:

        self.w = np.zeros(X.shape[1])
        prev_loss = float('inf')

        for _ in range(self.iters):
            y_pred = X @ self.w
            error = y_pred - y
            loss = np.mean(error ** 2)

            self.loss_history.append(loss)

            if abs(prev_loss - loss) < self.stop_iter:
                break

            grad = (X.T @ error) / X.shape[0]
            self.w -= self.lr * grad

            prev = loss

    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray] = None) -> "LinReg":

        X = self._preproc_data(X)
        y = y.values if isinstance(y, pd.Series) else y

        if self.solver == "close":
            self._fit_close(X, y)
        else:
            self._fit_gradient(X, y)

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        X = self._preproc_data(X)
        return X @ self.w

    def get_features_imp(self) -> pd.Series:
        pass
