import time
from contextlib import contextmanager

import pandas as pd


def load_data(file):
    df = pd.read_csv(file, low_memory=False)
    return df


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.0f} s")
