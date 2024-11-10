from v1.src.base.activation import *
from abc import ABC, abstractmethod

from v1.src.base.optimizers.optimizer import Optimizer
from v1.src.base.value_initializer import uniform_initializer, ValueInitializer, direct_assign_initializer


class Layer(ABC):
    pass
    def __init__(
            self,
            neurons: int,
            activation: Activation = None,
            is_trainable: bool = True,
            prev_weights_initializer: ValueInitializer = None,
            name: str = "ABCLayer",
    ):
        self.neurons = neurons
        self.activation = activation
        self.prev_weights_initializer = prev_weights_initializer
        self.is_trainable = is_trainable

        self.name = name

        self.prev_weights = None

    @abstractmethod
    def forward(self, in_batch: np.array, training = True) -> np.array:
        pass

    @abstractmethod
    def backward(self, next_layer_gradient_batch: np.array, optimizer: Optimizer) -> np.array:
        pass

    def init_layers_params(self, prev_layer_neurons, reassign_existing=True):
        if reassign_existing or self.prev_weights is None:
            self.prev_weights = self.prev_weights_initializer((prev_layer_neurons, self.neurons))

    def __getstate__(self):
        state = {
            'name': self.name,
            'neurons': self.neurons,
            'activation': self.activation,
            'is_trainable': self.is_trainable,
            'prev_weights': self.prev_weights,
        }
        return state

    def __setstate__(self, state):
        self.name = state['name']
        self.neurons = state['neurons']
        self.activation = state['activation']
        self.is_trainable = state['is_trainable']
        self.prev_weights = state['prev_weights']
        self.prev_weights_initializer = direct_assign_initializer(state['prev_weights'])
