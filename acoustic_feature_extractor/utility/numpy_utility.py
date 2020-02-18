from pathlib import Path
from typing import Union, Dict

import numpy


def load_numpy_object(path: Union[Path, str]) -> Dict[str, numpy.ndarray]:
    return numpy.load(str(path), allow_pickle=True).item()
