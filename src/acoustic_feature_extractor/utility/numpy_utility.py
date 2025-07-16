from pathlib import Path

import numpy


def load_numpy_object(path: Path | str) -> dict[str, numpy.ndarray]:
    return numpy.load(str(path), allow_pickle=True).item()
