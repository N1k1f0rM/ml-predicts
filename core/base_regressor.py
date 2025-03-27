from .base_predictor import BasePredictor
from abc import abstractmethod
import pandas as pd


class BaseRegressor(BasePredictor):

    @abstractmethod
    def get_features_imp(self) -> pd.Series:
        pass
