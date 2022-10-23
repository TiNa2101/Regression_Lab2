import pickle
from typing import Any, Union
from pandas import DataFrame


def save_as_pickle(obj: Any, path: str) -> None:
    if isinstance(obj, DataFrame):
        obj.to_pickle(path)
    else:
        with open(path, 'wb') as f:
            pickle.dump(obj, f)


def load_pickle(path: str) -> None:
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model
