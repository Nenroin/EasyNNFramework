from v1.src.base.loss_function import LossFunction, mse
from v1.src.base.metrics.metric import Metric


class LossMetric(Metric):
    def __init__(
            self,
            loss_function: LossFunction = None,
    ):
        super().__init__(
            name=type(self).__name__
        )
        self.__loss_function = loss_function

        self.iterations = 0
        self.overall_loss = 0.

        self.last_iterations = 0
        self.last_overall_loss = 0.

    @property
    def average_loss(self):
        return self.overall_loss / self.iterations

    def reset_state(self):
        self.last_iterations = self.iterations
        self.last_overall_loss = self.overall_loss

        self.iterations = 0
        self.overall_loss = 0.

    def update_state(self, y_pred, e):
        self.iterations += 1
        self.overall_loss += self.__loss_function(y_pred=y_pred, e=e)

    def print_result(self):
        print(f"iterations: {self.iterations}, overall_loss: {self.overall_loss}")
        print(f"average_loss: {self.average_loss}")


def average_mse_loss_metric():
    return LossMetric(loss_function=mse())