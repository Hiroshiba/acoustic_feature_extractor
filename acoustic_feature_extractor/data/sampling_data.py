from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Union

import numpy


@dataclass
class SamplingData:
    array: numpy.ndarray  # shape: (N, ?)
    rate: float

    def resample(self, sampling_rate: float, index: int, length: int):
        assert sampling_rate % self.rate == 0, f"{sampling_rate} {self.rate}"
        scale = int(sampling_rate // self.rate)

        ni = index // scale
        nl = length // scale + 2

        array = self.array[ni : ni + nl]
        if scale > 1:
            array = numpy.repeat(array, scale, axis=0)

        i = index - ni * scale
        return array[i : i + length]

    def split(
        self,
        keypoint_seconds: Union[Sequence[float], numpy.ndarray],
    ):
        keypoint_seconds = numpy.array(keypoint_seconds)
        indexes = (keypoint_seconds * self.rate).astype(numpy.int32)
        arrays = numpy.split(self.array, indexes)
        return [self.__class__(array=array, rate=self.rate) for array in arrays]

    def estimate_padding_value(self):
        values = numpy.concatenate((self.array[:5], self.array[-5:]), axis=0)
        assert len(values) > 0

        value = values[0]
        for i in range(1, len(values)):
            assert numpy.all(value == values[i])

        return value[numpy.newaxis]

    @staticmethod
    def padding(datas: Sequence["SamplingData"], padding_value: numpy.ndarray):
        datas = deepcopy(datas)

        max_length = max(len(d.array) for d in datas)
        for data in datas:
            padding_array = padding_value.repeat(max_length - len(data.array), axis=0)
            data.array = numpy.concatenate([data.array, padding_array])

        return datas

    def all_same(self):
        value = self.array[0][numpy.newaxis]
        return numpy.all(value == self.array)

    @staticmethod
    def collect(
        datas: Sequence["SamplingData"], rate: int, mode: str, error_time_length: float
    ):
        scales = [(rate // d.rate) for d in datas]
        arrays: Sequence[numpy.ndarray] = [
            d.resample(sampling_rate=rate, index=0, length=int(len(d.array) * s))
            for d, s in zip(datas, scales)
        ]

        # assert that nearly length
        max_length = max(len(a) for a in arrays)
        for i, a in enumerate(arrays):
            assert (
                abs((max_length - len(a)) / rate) <= error_time_length
            ), f"{i}: {max_length / rate}, {len(a) / rate}"

        if mode == "min":
            min_length = min(len(a) for a in arrays)
            array = numpy.concatenate([a[:min_length] for a in arrays], axis=1).astype(
                numpy.float32
            )
        elif mode == "max":
            arrays = [
                numpy.pad(a, ((0, max_length - len(a)), (0, 0)))
                if len(a) < max_length
                else a
                for a in arrays
            ]
            array = numpy.concatenate(arrays, axis=1).astype(numpy.float32)
        else:
            raise ValueError(mode)

        return array

    @classmethod
    def load(cls, path: Path):
        d: Dict = numpy.load(str(path), allow_pickle=True).item()
        array, rate = d["array"], d["rate"]

        if array.ndim == 1:
            array = array[:, numpy.newaxis]

        return cls(array=array, rate=rate)

    def save(self, path: Path):
        numpy.save(str(path), dict(array=self.array, rate=self.rate))
