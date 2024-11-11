from abc import abstractmethod


class Metric:
    def __init__(
            self,
            name: str,
    ):
        self.name = name

    @abstractmethod
    def update_state(self, y_pred, e):
        pass

    @abstractmethod
    def clear_state(self):
        pass

    @abstractmethod
    def get_metric_state(self):
        pass

    @abstractmethod
    def get_metric_value(self):
        pass