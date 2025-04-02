from core.base_classifier import BaseClassifier
from typing import Literal, List, Union, Optional, Self
import numpy as np


class LogReg(BaseClassifier):

    def __init__(self, gd_type: Literal["stochastic", "classic"] = "classic",
                 tolerance: float = 1e-4,
                 max_iter: int = 1000,
                 w0: Optional[np.ndarray] = None,
                 eta: float = 1e-2) -> None:

        self.gd_type = gd_type
        self.tol = tolerance
        self.max_iter = max_iter
        self.w0 = w0
        self.w: Optional[List[float]] = None
        self.eta = eta
        self.loss_history: Optional[List[float]] = None
        self.__weights_history: List[float] = []

    @property
    def get_weights(self):
        return self.__weights_history

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:

        if self.w0 is None:
            self.w = np.zeros(X.shape[1])
        else:
            self.w = self.w0

        self.loss_history = []

        self.calc_gradient(X, y)

        final_loss = self.calc_loss(X, y)
        print(f"Final loss after training: {final_loss}")

        return self

    def predict(self, X: np.ndarray):
        if self.w is None:
            raise Exception('Not trained yet')

        return (np.dot(X, self.w) > 0).astype(int)

    def predict_proba(self, X: np.ndarray):
        if self.w is None:
            raise Exception('Not trained yet')

        return 1 / (1 + np.exp(-np.dot(X, self.w)))

    @staticmethod
    def __sigmoid(z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def _loss_grad(self, X: np.ndarray, y: np.ndarray,
                   w: np.ndarray) -> np.ndarray:

        predictions = self.__sigmoid(np.dot(X, w))
        gradient = -np.dot(X.T, (y - predictions)) / len(y)
        return gradient

    def _stoch_loss_grad(self, X: np.ndarray, y: np.ndarray,
                         w: np.ndarray) -> np.ndarray:

        predictions = self.__sigmoid(np.dot(X, w))
        gradient = -np.dot(X.T, (y - predictions))
        return gradient

    def calc_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:

        self.w = self.w0
        prev_w = self.w

        self.__weights_history.append(self.w)

        dim = X.shape[0]

        match self.gd_type:
            case "classic":
                for i in range(self.max_iter):
                    loss_grad = self._loss_grad(X, y, self.w)
                    self.w = self.w - self.eta * loss_grad
                    self.__weights_history.append(self.w)

                    if ((np.linalg.norm(self.w - prev_w) < self.tol)
                       and (self.max_iter == i)):
                        break

                    previous_w = self.w

                return self.w

            case "stochastic":
                for i in range(self.max_iter):
                    st = np.random.randint(0, dim)
                    loss_grad = self._stoch_loss_grad(X[st].reshape(1, -1),
                                                      y[st], self.w)
                    self.w = self.w - self.eta * loss_grad
                    self.__weights_history.append(self.w)

                    if ((np.linalg.norm(self.w - prev_w) < self.tol)
                       and (self.max_iter == i)):
                        break

                    previous_w = self.w

                return self.w

            case _:
                raise Exception("Wrong type, use full or stochastic")

    def calc_loss(self, X: np.ndarray, y: np.ndarray) -> float:

        self.loss_history = []

        if self.w is None:
            self.w = np.zeros(X.shape[1])

        for i in self.__weights_history:
            loss = -np.mean(y * np.log(self.__sigmoid(np.dot(X, i))+ 1e-18) +
                   (1 - y) * np.log(1 - self.__sigmoid(np.dot(X, i))+ 1e-18))

            self.loss_history.append(loss)

        return self.loss_history[-1]
