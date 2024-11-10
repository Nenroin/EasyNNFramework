from typing import Callable
import numpy as np

from v1.src.serialize.serialize_meta_class import SerializeMetaClass


class LossFunction(
    metaclass=SerializeMetaClass,
    id_field='name',
    saved_fields = ['__loss_fn', '__gradient_fn'],
):
    def __init__(self,
                 loss_fn: Callable[[np.array, np.array], np.array],
                 gradient_fn: Callable[[np.array, np.array], np.array],
                 name: str
                 ):
        self.__loss_fn = loss_fn
        self.__gradient_fn = gradient_fn

        self.name = name

    def __call__(self, y_pred: np.array, e: np.array) -> np.array:
        return self.__loss_fn(y_pred, e)

    def gradient(self, y_pred: np.array, e: np.array) -> np.array:
        return self.__gradient_fn(y_pred, e)


def mse() -> LossFunction:
    def call(y_pred: np.array, e: np.array) -> np.array:
        return np.power(y_pred - e, 2).sum() / 2

    def gradient(y_pred: np.array, e: np.array) -> np.array:
        return y_pred - e

    return LossFunction(call, gradient, "mse_loss")

LossFunction.__register_generators__([mse])