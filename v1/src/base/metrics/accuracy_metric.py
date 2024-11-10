from v1.src.base.metrics.matching_functon import MatchingFunction, one_hot_matching_function
from v1.src.base.metrics.metric import Metric


class AccuracyMetric(Metric):
    def __init__(
            self,
            matching_function: MatchingFunction = None,
    ):
        super().__init__(
            name=type(self).__name__
        )
        self.__matching_function = matching_function

        self.iterations = 0
        self.guessed_counter = 0

        self.last_iterations = 0
        self.last_guessed_counter = 0.

    @property
    def accuracy(self):
        return self.guessed_counter / self.iterations * 100

    def reset_state(self):
        self.last_iterations = self.iterations
        self.last_guessed_counter = self.guessed_counter

        self.iterations = 0
        self.guessed_counter = 0

    def update_state(self, y_pred, e):
        self.iterations += 1
        self.guessed_counter += (self.__matching_function(y_pred, e))

    def print_result(self):
        print(f"iterations: {self.iterations}, guessed_counter: {self.guessed_counter}")
        print(f"accuracy: {self.accuracy}")


def one_hot_matches_metric():
    return AccuracyMetric(matching_function=one_hot_matching_function())