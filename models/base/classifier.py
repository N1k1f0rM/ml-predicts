from .predictor import BasePredictor
from abc import abstractmethod
import pandas as pd
import numpy as np
from typing import Union


class BaseClassifier(BasePredictor):

    @abstractmethod
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        pass
