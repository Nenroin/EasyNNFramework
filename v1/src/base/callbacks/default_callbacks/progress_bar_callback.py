import sys
from collections import ChainMap

from v1.src.base.callbacks import Callback
from v1.src.utils import ProgressBar


class ProgressBarCallback(Callback):
    def __init__(
            self,
            monitors: [str] = None,
            # count_mode = 'batch' or 'sample'
            count_mode: str = 'batch'
    ):
        super().__init__()

        self.monitors = monitors or ["average_loss", "accuracy"]
        self.count_mode = _check_count_mode(count_mode)
        self.progress_bar: ProgressBar = None

        self.epochs = 1
        self.test_batches = 1
        self.train_batches = 1
        self.batch_size = 100

        self.out_stream = sys.stderr

    @property
    def test_total(self):
        return self.test_batches if self.count_mode == 'batch' else self.test_batches * self.batch_size

    @property
    def train_total(self):
        return self.train_batches if self.count_mode == 'batch' else self.train_batches * self.batch_size

    @property
    def update_step(self):
        return 1 if self.count_mode == 'batch' else self.batch_size

    def set_state(self, state):
        super().set_state(state)
        self.epochs = state.get('epochs', self.epochs)
        self.test_batches = state.get('test_batches', self.test_batches)
        self.train_batches = state.get('train_batches', self.train_batches)
        self.batch_size = state.get('batch_size', self.batch_size)

    def on_epoch_start(self, epoch, state_dict=None):
        print(f"Epoch: {epoch+1}/{self.epochs}", file=self.out_stream)

    def on_train_epoch_start(self, epoch, state_dict=None):
        self.progress_bar = ProgressBar(
            total=self.train_total,
            prefix="train: "
        )

    def on_train_batch_end(self, batch_idx, metric_states, state_dict=None):
        merged_metric_dict = dict(ChainMap(*metric_states))
        self.__update_progress_bar_postfix(merged_metric_dict)
        self.progress_bar.update(increase=self.update_step)

    def on_train_epoch_end(self, epoch, metric_states, state_dict=None):
        self.progress_bar.close()
        self.progress_bar = None

    def on_test_epoch_start(self, epoch, state_dict=None):
        self.progress_bar = ProgressBar(
            total=self.test_total,
            prefix="test:  "
        )

    def on_test_batch_end(self, batch_idx, metric_states, state_dict=None):
        merged_metric_dict = dict(ChainMap(*metric_states))
        self.__update_progress_bar_postfix(merged_metric_dict)
        self.progress_bar.update(increase=self.update_step)

    def on_test_epoch_end(self, epoch, metric_states, state_dict=None):
        self.progress_bar.close()
        self.progress_bar = None

    def __update_progress_bar_postfix(self, metric_dict):
        monitor_items = {monitor: metric_dict[monitor]
                         for monitor in self.monitors
                         if monitor in metric_dict}
        monitors_str = ", ".join(f"{key}: {value:.3f}" for key, value in monitor_items.items())
        self.progress_bar.set_postfix(monitors_str)

def _check_count_mode(count_mode):
    allowed_values = ['batch', 'sample']
    if count_mode not in allowed_values:
        raise ValueError(
            "Invalid value for argument 'count_mode'. "
            f"Allowed {allowed_values}. "
            f"Received: {count_mode}."
        )
    else:
        return count_mode