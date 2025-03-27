from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Union


class BaseModels(ABC):

    @abstractmethod
    def fit(self, X: Union[pd.DataFrame, np.ndarray],
                  y : Union[pd.Series, np.ndarray] = None) -> None:
        pass
