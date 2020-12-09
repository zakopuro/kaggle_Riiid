import re
import time
from abc import ABCMeta, abstractmethod
from pathlib import Path
from contextlib import contextmanager
import argparse
import inspect
import pandas as pd


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', '-f', action='store_true', help='Overwrite existing files')
    return parser.parse_args()


def get_features(namespace):
    for k, v in namespace.items():
        if inspect.isclass(v) and issubclass(v, Feature) and not inspect.isabstract(v):
            yield v()


def generate_features(namespace, overwrite):
    for f in get_features(namespace):
        if f.train_path.exists() and not overwrite:
            print(f.name, 'was skipped')
        else:
            f.run().save()

@contextmanager
def timer(name):
    t0 = time.time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


class Feature(metaclass=ABCMeta):
    prefix = ''
    suffix = ''
    dir = '.'

    def __init__(self):
        self.name = self.__class__.__name__
        self.train = pd.DataFrame()
        self.valid = pd.DataFrame()
        self.test = pd.DataFrame()
        self.train_path = Path(self.dir) / f'{self.name}_train.feather'
        self.valid_path = Path(self.dir) / f'{self.name}_valid.feather'
        # self.test_path = Path(self.dir) / f'kaggle_kernel/{self.name}_test.feather'
        print(self.train_path)

    def run(self):
        with timer(self.name):
            self.create_features()
            prefix = self.prefix + '_' if self.prefix else ''
            suffix = '_' + self.suffix if self.suffix else ''
            self.train.columns = prefix + self.train.columns + suffix
            # self.test.columns = prefix + self.test.columns + suffix
            self.valid.columns = prefix + self.valid.columns + suffix
        return self

    @abstractmethod
    def create_features(self):
        raise NotImplementedError

    def save(self):
        a = 1
        # self.train.to_feather(str(self.train_path))
        # self.valid.to_feather(str(self.valid_path))
        # self.test.to_feather(str(self.test_path))
