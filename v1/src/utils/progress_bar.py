import sys
import time
from typing import TextIO

class ProgressBar:
    def __init__(
            self,
            total: int = None,
            pb_length: int = 50,
            fill: chr = '█',
            out_stream: TextIO = None,
            prefix: str = '',
            suffix: str = '',
            unit: str = 'it',
            initial: int = 0,
    ):
        if out_stream is None:
            out_stream = sys.stderr

        self.pb_length = pb_length
        self.fill = fill
        self.prefix = prefix
        self.suffix = suffix
        self.unit = unit
        self.total = total
        self.out_stream = out_stream
        self.initial = initial

        self.iteration = initial

        self.start_time = time.time()
        self.last_iteration_change_time = None
        self.last_print_time = None

    def update(self, increase=1):
        if not isinstance(increase, int) or increase <= 0:
            raise ValueError('Increase must be an integer greater than 0.')
        self.iteration += increase
        self.last_iteration_change_time = time.time()
        self.refresh()

    def refresh(self):
        bar = self.__get_progress_bar()
        time_section = self.__get_time_section()
        iterations_section = self.__get_iterations_section()
        completion_percentage = self.__get_completion_percentage()
        self.display(f'{self.prefix} {completion_percentage} |{bar}| {iterations_section} {time_section} {self.suffix}')

    def display(self, msg=None):
        self.last_print_time = self.start_time
        self.out_stream.write(f'\r{msg}')
        self.out_stream.flush()

    def __get_progress_bar(self):
        iteration = min(self.iteration, self.total)
        filled_length = int(self.pb_length * iteration // self.total)
        bar = self.fill * filled_length + '-' * (self.pb_length - filled_length)
        return bar

    def __get_time_section(self):
        time_to_finish = self.__get_time_to_finish()
        displayed_time_to_finish = self.__format_time(time_to_finish)
        passed_time = self.__get_passed_time()
        displayed_passed_time = self.__format_time(passed_time)
        return f'[{displayed_passed_time} < {displayed_time_to_finish}]'

    def __get_iterations_section(self):
        total_digits_number = len(str(self.total))
        return f'{self.iteration: {total_digits_number}}{self.unit}/{self.total}{self.unit}'

    def __get_completion_percentage(self):
        percentage = int(self.iteration / self.total * 100)
        return f'{percentage: 3}%'

    @classmethod
    def __format_time(cls, seconds):
        if seconds >= 3600:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            formatted_time = f"{hours:02}h{minutes:02}m"
        elif seconds >= 60:
            minutes = int(seconds // 60)
            seconds = int(seconds % 60)
            formatted_time = f"{minutes:02}m{seconds:02}s"
        else:
            milliseconds = int((seconds % 1) * 100)
            seconds = int(seconds)
            formatted_time = f"{seconds:02}s{milliseconds:03}ms"

        return formatted_time

    def __get_time_to_finish(self):
        remaining_iterations = self.total - self.iteration
        remaining_time = self.__get_average_iteration_duration() * remaining_iterations
        return remaining_time

    def __get_passed_time(self):
        return self.last_iteration_change_time - self.start_time

    def __get_average_iteration_duration(self):
        return self.__get_passed_time() / (self.iteration - self.initial)
