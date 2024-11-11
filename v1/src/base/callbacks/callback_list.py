from v1.src.base.callbacks import Callback
from v1.src.base.callbacks.callback_interface import _CallbackInterface
from v1.src.utils import is_overridden


class CallbackList(_CallbackInterface):
    def __init__(
            self,
            model=None,
            epochs: int = None,
            train_iterations: int = None,
            test_iterations: int = None,
            callbacks: [Callback] = None,
            **kwargs
    ):
        self.model = model
        self.__callbacks = callbacks or []

        self.epochs = epochs
        self.train_iterations = train_iterations
        self.test_iterations = test_iterations

        self.recompile_callbacks(**kwargs)

        self.callbacks_mapping = self.__get_callbacks_mapping()

    def update_params(
            self,
            model=None,
            epochs: int = None,
            train_iterations: int = None,
            test_iterations: int = None,
            **kwargs
    ):
        self.model = model or self.model

        self.epochs = epochs or self.epochs
        self.train_iterations = train_iterations or self.train_iterations
        self.test_iterations = test_iterations or self.test_iterations

        self.recompile_callbacks(**kwargs)

    def recompile_callbacks(self, **kwargs):
        [callback.set_model(self.model)
         for callback in self.__callbacks]

        [callback.update_state(
            {
                'epochs': self.epochs,
                'train_iterations': self.train_iterations,
                'test_iterations': self.test_iterations,
                **kwargs,
            }
        ) for callback in self.__callbacks]

    def __get_callbacks_mapping(self):
        callbacks_mapping = {
            fn_name: self.__extract_functions_from_callbacks(fn_name)
            for fn_name in _CallbackInterface.get_callback_names()
        }
        return callbacks_mapping

    def __extract_functions_from_callbacks(self, function_name):
        return [getattr(callback, function_name)
                for callback
                in self.__callbacks
                if is_overridden(getattr(callback, function_name))]

    def on_fit_start(self, epochs, state_dict=None):
        [callback(epochs, state_dict)
         for callback in self.callbacks_mapping['on_fit_start']]

    def on_fit_end(self, state_dict=None):
        [callback(state_dict)
         for callback in self.callbacks_mapping['on_fit_end']]

    def on_epoch_start(self, epoch, state_dict=None):
        [callback(epoch, state_dict)
         for callback in self.callbacks_mapping['on_epoch_start']]

    def on_epoch_end(self, epoch, state_dict=None):
        [callback(epoch, state_dict)
         for callback in self.callbacks_mapping['on_epoch_end']]

    def on_train_epoch_start(self, epoch, state_dict=None):
        [callback(epoch, state_dict)
         for callback in self.callbacks_mapping['on_train_epoch_start']]

    def on_train_epoch_end(self, epoch, metric_state, state_dict=None):
        [callback(epoch, metric_state, state_dict)
         for callback in self.callbacks_mapping['on_train_epoch_end']]

    def on_test_epoch_start(self, epoch, state_dict=None):
        [callback(epoch, state_dict)
         for callback in self.callbacks_mapping['on_test_epoch_start']]

    def on_test_epoch_end(self, epoch, metric_state, state_dict=None):
        [callback(epoch, metric_state, state_dict)
         for callback in self.callbacks_mapping['on_test_epoch_end']]

    def on_train_batch_start(self, batch_idx, state_dict=None):
        [callback(batch_idx, state_dict)
         for callback in self.callbacks_mapping['on_train_batch_start']]

    def on_train_batch_end(self, batch_idx, metric_state, state_dict=None):
        [callback(batch_idx, metric_state, state_dict)
         for callback in self.callbacks_mapping['on_train_batch_end']]

    def on_test_batch_start(self, batch_idx, state_dict=None):
        [callback(batch_idx, state_dict)
         for callback in self.callbacks_mapping['on_test_batch_start']]

    def on_test_batch_end(self, batch_idx, metric_state, state_dict=None):
        [callback(batch_idx, metric_state, state_dict)
         for callback in self.callbacks_mapping['on_test_batch_end']]


    # def on_train_forward_start(self, batch_idx, state_dict=None):
    #     [callback(batch_idx, state_dict)
    #      for callback in self.callbacks_mapping['on_train_forward_start']]
    #
    # def on_train_forward_end(self, batch_idx, state_dict=None):
    #     [callback(batch_idx, state_dict)
    #      for callback in self.callbacks_mapping['on_train_forward_end']]
    #
    # def on_test_forward_start(self, batch_idx, state_dict=None):
    #     [callback(batch_idx, state_dict)
    #      for callback in self.callbacks_mapping['on_test_forward_start']]
    #
    # def on_test_forward_end(self, batch_idx, state_dict=None):
    #     [callback(batch_idx, state_dict)
    #      for callback in self.callbacks_mapping['on_test_forward_end']]
    #
    # def on_train_backward_start(self, loss_gradient, state_dict=None):
    #     [callback(loss_gradient, state_dict)
    #      for callback in self.callbacks_mapping['on_train_backward_start']]
    #
    # def on_train_backward_end(self, state_dict=None):
    #     [callback(state_dict)
    #      for callback in self.callbacks_mapping['on_train_backward_end']]