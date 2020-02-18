from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy


@dataclass
class SamplingData:
    array: numpy.ndarray  # shape: (N, ?)
    rate: float

    def resample(self, sampling_rate: float, index: int, length: int):
        assert sampling_rate % self.rate == 0, f'{sampling_rate} {self.rate}'
        scale = int(sampling_rate // self.rate)

        ni = index // scale
        nl = length // scale + 2

        array = self.array[ni:ni + nl]
        if scale > 1:
            array = numpy.repeat(array, scale, axis=0)

        i = index - ni * scale
        return array[i:i + length]

    @classmethod
    def load(cls, path: Path):
        d: Dict = numpy.load(str(path), allow_pickle=True).item()
        array, rate = d['array'], d['rate']

        if array.ndim == 1:
            array = array[:, numpy.newaxis]

        return cls(
            array=array,
            rate=rate,
        )

    def save(self, path: Path):
        numpy.save(str(path), dict(array=self.array, rate=self.rate))
