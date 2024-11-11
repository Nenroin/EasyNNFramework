from typing import Callable

import numpy as np

from v1.src.serialize import SerializeMetaClass


class Activation(
    metaclass=SerializeMetaClass,
    id_field='name',
    saved_fields = ['__activation_fn', '__jacobian_fn'],
):
    def __init__(self,
                 activation_fn: Callable[[np.array], np.array],
                 jacobian_fn: Callable[[np.array], np.array],
                 name: str,
                 ):
        self.__activation_fn = activation_fn
        self.__jacobian_fn = jacobian_fn

        self.name = name

    def __call__(self, x_vec: np.array) -> np.array:
        return self.__activation_fn(x_vec)

    def jacobian(self, x_vec: np.array) -> np.array:
        return self.__jacobian_fn(x_vec)

def relu() -> Activation:
    def call(x: np.array) -> np.array:
        return np.maximum(x, 0)

    def jacobian(x: np.array) -> np.array:
        x = np.array(x)
        jacobian__ = np.diag(np.where(x <= 0, 0, 1))
        return jacobian__

    return Activation(call, jacobian, "relu")

def linear() -> Activation:
    def call(x: np.array) -> np.array:
        return x

    def jacobian(x: np.array) -> np.array:
        x = np.array(x)
        jacobian_ = np.diag(np.ones(len(x)))
        return jacobian_

    return Activation(call, jacobian, "linear")

def softmax() -> Activation:
    def call(s: np.array) -> np.array:
        z = s - s.max()
        return np.exp(z) / sum(np.exp(z))

    def jacobian(s: np.array) -> np.array:
        jacobian_matrix = np.diagflat(s) + np.einsum('i,j->ij',
                                          s, s,
                                          optimize='optimal')
        return jacobian_matrix

    return Activation(call, jacobian, "softmax")

def sigmoid() -> Activation:
    def call(x: np.array) -> np.array:
        return np.array(1 / (1 + np.exp(-x)))

    def jacobian(x: np.array) -> np.array:
        sigmoids = call(x)
        jacobian_ = np.diag(np.multiply(sigmoids, (1 - sigmoids)))
        return jacobian_

    return Activation(call, jacobian, "sigmoid")

def arctan() -> Activation:
    def call(x: np.array) -> np.array:
        return np.arctan(x)

    def jacobian(x: np.array) -> np.array:
        jacobian_ = np.diag(np.array([1 / (1 + xi**2) for xi in x]))
        return jacobian_

    return Activation(call, jacobian, "arctan")

Activation.__register_generators__([relu, arctan, linear, softmax, sigmoid, sigmoid])
