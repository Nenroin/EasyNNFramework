from abc import ABC, abstractmethod


class _CallbackInterface(ABC):
    @staticmethod
    def get_callback_names():
        return [func for func in dir(_CallbackInterface) if func.startswith('on_')]

    @abstractmethod
    def on_fit_start(self, epochs, state_dict=None):
        pass

    @abstractmethod
    def on_fit_end(self, state_dict=None):
        pass

    @abstractmethod
    def on_epoch_end(self, epoch, state_dict=None):
        pass

    @abstractmethod
    def on_epoch_start(self, epoch, state_dict=None):
        pass

    @abstractmethod
    def on_train_epoch_start(self, epoch, state_dict=None):
        pass

    @abstractmethod
    def on_train_epoch_end(self, epoch, metric_state, state_dict=None):
        pass

    @abstractmethod
    def on_test_epoch_start(self, epoch, state_dict=None):
        pass

    @abstractmethod
    def on_test_epoch_end(self, epoch, metric_state, state_dict=None):
        pass

    @abstractmethod
    def on_train_batch_start(self, batch_idx, state_dict=None):
        pass

    @abstractmethod
    def on_train_batch_end(self, batch_idx, metric_state, state_dict=None):
        pass

    @abstractmethod
    def on_test_batch_start(self, batch_idx, state_dict=None):
        pass

    @abstractmethod
    def on_test_batch_end(self, batch_idx, metric_state, state_dict=None):
        pass

    # @abstractmethod
    # def on_train_forward_start(self, batch_idx, metric_state, state_dict=None):
    #     pass
    #
    # @abstractmethod
    # def on_train_forward_end(self, batch_idx, metric_state, state_dict=None):
    #     pass
    #
    # @abstractmethod
    # def on_test_forward_start(self, batch_idx, metric_state, state_dict=None):
    #     pass
    #
    # @abstractmethod
    # def on_test_forward_end(self, batch_idx, metric_state, state_dict=None):
    #     pass
    #
    # @abstractmethod
    # def on_train_backward_start(self, batch_idx, metric_state, state_dict=None):
    #     pass
    #
    # @abstractmethod
    # def on_train_backward_end(self, batch_idx, metric_state, state_dict=None):
    #     pass