from .models import BaseModels
import pandas as pd
import numpy as np
from typing import Union
from abc import abstractmethod


class BaseTransformer(BaseModels):

    @abstractmethod
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        pass


    @abstractmethod
    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray],
                            y: Union[pd.Series, np.ndarray] = None) -> Union[pd.DataFrame, np.ndarray]:
        pass
