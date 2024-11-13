import time
from time import sleep

from tqdm import tqdm

from v1.src.utils import ProgressBar

progress_bar = ProgressBar(
    pb_length= 30,
    fill = '█',
    out_stream= None,
    prefix = '',
    suffix = '',
    unit = 'it',
    total = 10,
)

for i in [1,2,3,4,5,6,7,8,9,10]:
    sleep(1.15)
    progress_bar.update(1)
