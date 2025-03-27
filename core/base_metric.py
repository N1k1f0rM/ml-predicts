from abc import abstractmethod


class BaseMetric:

    @abstractmethod
    def loss(self, ):
        pass