from abc import ABC

import numpy as np

from v1.src.base.callbacks.callback_list import CallbackList
from v1.src.base.data import ModelDataSource
from v1.src.base.layers.layer import Layer
from v1.src.base.loss_function import LossFunction
from v1.src.base.metrics.metric import Metric
from v1.src.base.models.model import Model
from v1.src.base.optimizers.optimizer import Optimizer
from v1.src.utils.sequential_model_summary_util import print_model_summary


class SequentialModel(Model, ABC):
    def __init__(self,
                 layers: [Layer] = None,
                 optimizer: Optimizer = None,
                 loss_function: LossFunction = None,
                 metric: Metric = None,
                 name: str = 'SequentialModel',
                 ):
        super().__init__(
            layers=layers,
            optimizer=optimizer,
            loss_function=loss_function,
            metric=metric,
            name=name,
        )

    def fit(self,
            model_data_source: ModelDataSource,
            epochs: int = 1,
            callbacks: CallbackList = None):
        train_data = model_data_source.train_data_batches()
        test_data = model_data_source.test_data_batches()
        for epoch in range(epochs):
            self.train_epoch(train_data)
            self.test_epoch(test_data)

    def forward(self, in_batch: np.ndarray, training=True) -> np.ndarray:
        if in_batch.ndim == 1:
            in_batch = np.expand_dims(in_batch, axis=0)

        for layer in self.layers[1:]:
            in_batch = layer.forward(in_batch)
        return in_batch

    def backward(self, loss_gradient_batch: np.array):
        if loss_gradient_batch.ndim == 1:
            loss_gradient_batch = np.expand_dims(loss_gradient_batch, axis=0)

        for layer in reversed(self.layers[1:]):
            loss_gradient_batch = layer.backward(loss_gradient_batch, optimizer=self.optimizer)

    def add_layer(self, layer: Layer):
        layer.init_layers_params(
            prev_layer_neurons=self.layers[-1].neurons,
            reassign_existing=True
        )
        self.layers.append(layer)

    def init_layers_params(self, reassign_existing=True):
        for i in range(1, len(self.layers)):
            self.layers[i].init_layers_params(
                prev_layer_neurons=self.layers[i-1].neurons,
                reassign_existing=reassign_existing
            )

    def summary(self):
        print_model_summary(self)