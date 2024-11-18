from typing import Callable
import numpy as np


class ValueInitializer:
    def __init__(
            self,
            init_function: Callable[[np.array], np.array],
    ):
        self.__init_function = init_function

    def __call__(self, shape):
        create_array = self.__init_function(shape)

        if create_array.ndim == 1:
            create_array = np.expand_dims(create_array, axis=0)

        return create_array

def zero_initializer() -> ValueInitializer:
    def zero_init_function(shape) -> np.array:
        return np.zeros(shape)

    return ValueInitializer(zero_init_function)

def uniform_initializer(min_value: float = -0.5, max_value: float = 0.5) -> ValueInitializer:
    def uniform_init_function(shape) -> np.array:
        return np.random.uniform(min_value, max_value, shape)

    return ValueInitializer(uniform_init_function)

def none_weights_initializer() -> ValueInitializer:
    def none_weights_init_function(_) -> np.array:
        return None

    return ValueInitializer(none_weights_init_function)

def he_initializer() -> ValueInitializer:
    def he_init_function(shape) -> np.array:
        n_in, n_out = shape
        return np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)

    return ValueInitializer(he_init_function)

def xavier_initializer() -> ValueInitializer:
    def xavier_init_function(shape) -> np.array:
        n_in, n_out = shape
        return np.random.randn(n_in, n_out) * np.sqrt(1 / (n_in + n_out))

    return ValueInitializer(xavier_init_function)

def orthogonal_initializer() -> ValueInitializer:
    def orthogonal_init_function(shape) -> np.array:
        n_in, n_out = shape
        return np.linalg.qr(np.random.randn(n_in, n_out))[0]

    return ValueInitializer(orthogonal_init_function)

def direct_assign_initializer(weights: np.array) -> ValueInitializer:
    def direct_init_function(shape) -> np.array:
        if weights.shape != shape:
            raise ValueError("wrong dimensions")
        return weights

    return ValueInitializer(direct_init_function)