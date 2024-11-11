from v1.src.base.callbacks.callback_interface import _CallbackInterface
from v1.src.utils import not_overridden


class Callback(_CallbackInterface):
    def __init__(
            self,
    ):
        self.model = None
        self.state = {}

    def set_state(self, state):
        self.state = state

    def update_state(self, state):
        self.state = self.state.update(state)

    def set_model(self, model):
        self.model = model

    @not_overridden
    def on_fit_start(self, epochs, state_objdict=None):
        pass

    @not_overridden
    def on_fit_end(self, state_dict=None):
        pass

    @not_overridden
    def on_epoch_start(self, epoch, state_dict=None):
        pass

    @not_overridden
    def on_epoch_end(self, epoch, state_dict=None):
        pass

    @not_overridden
    def on_train_epoch_start(self, epoch, state_dict=None):
        pass

    @not_overridden
    def on_train_epoch_end(self, epoch, metric_state, state_dict=None):
        pass

    # metric_state - passed from Metric.get_metric_state(self)
    @not_overridden
    def on_test_epoch_start(self, epoch, state_dict=None):
        pass

    # metric_state - passed from Metric.get_metric_state(self)
    @not_overridden
    def on_test_epoch_end(self, epoch, metric_state, state_dict=None):
        pass
    
    @not_overridden
    def on_train_batch_start(self, batch, state_dict=None):
        pass

    @not_overridden
    def on_train_batch_end(self, batch, metric_state, state_dict=None):
        pass

    @not_overridden
    def on_test_batch_start(self, batch, state_dict=None):
        pass

    @not_overridden
    def on_test_batch_end(self, batch, metric_state, state_dict=None):
        pass

    # @not_overridden
    # def on_train_forward_start(self, batch_idx, state_dict=None):
    #     pass
    # 
    # @not_overridden
    # def on_train_forward_end(self, batch_idx, state_dict=None):
    #     pass
    # 
    # @not_overridden
    # def on_test_forward_start(self, batch_idx, state_dict=None):
    #     pass
    # 
    # @not_overridden
    # def on_test_forward_end(self, batch_idx, state_dict=None):
    #     pass
    # 
    # @not_overridden
    # def on_train_backward_start(self, loss_gradient, state_dict=None):
    #     pass
    # 
    # @not_overridden
    # def on_train_backward_end(self, state_dict=None):
    #     pass