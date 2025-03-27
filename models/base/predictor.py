from .models import BaseModels
from abc import abstractmethod
import pandas as pd
import numpy as np
from typing import Union

class BasePredictor(BaseModels):

    @abstractmethod
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        pass
    