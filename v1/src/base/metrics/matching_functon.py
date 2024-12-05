from typing import Callable

import numpy as np

from v1.src.serialize.serialize_meta_class import SerializeMetaClass


class MatchingFunction(
    metaclass=SerializeMetaClass,
    id_field='name',
    saved_fields = ['__matching_function'],
):
    def __init__(self,
                 name: str = "",
                 matching_fn: Callable[[np.array, np.array], bool] = None,
                 ):
        self.name = name

        self.__matching_function = matching_fn

    def __call__(self, y_pred: np.array, e: np.array) -> np.array:
        return self.__matching_function(y_pred, e)

def one_hot_matching_function():
    def matching_function(y_pred, e):
        return y_pred.argmax() == e.argmax()

    return MatchingFunction(matching_fn=matching_function,
                            name="one_hot_matching_function")

MatchingFunction.__register_generators__([one_hot_matching_function])