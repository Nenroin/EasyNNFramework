import numpy as np

from v1.src.base.activation import linear
from v1.src.base.optimizers.optimizer import Optimizer
from v1.src.base.value_initializer import none_weights_initializer
from v1.src.base.layers.layer import Layer


class InputLayer(Layer):
    def __init__(
            self,
            neurons: int,
    ):
        super().__init__(
            neurons=neurons,
            activation=linear(),
            prev_weights_initializer=none_weights_initializer(),
            is_trainable=False,
            name=self.__class__.__name__,
        )

    # Y = X * W
    def forward(self, in_batch: np.array, training=True) -> np.array:
        return in_batch

    def backward(self, layer_gradient_batch: np.array, optimizer: Optimizer):
        return layer_gradient_batch

