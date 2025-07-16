from pathlib import Path

import librosa
import numpy
import soundfile
from resampy import resample

from acoustic_feature_extractor.data.sampling_data import SamplingData


class Wave:
    def __init__(self, wave: numpy.ndarray, sampling_rate: int):
        self.wave = wave
        self.sampling_rate = sampling_rate

    @staticmethod
    def load(path: Path, sampling_rate: int = None, dtype=numpy.float32):
        if path.suffix == ".npy" or path.suffix == ".npz":
            a = SamplingData.load(path)
            a.array = numpy.squeeze(a.array)
            if sampling_rate is not None:
                a.array = resample(a.array, a.rate, sampling_rate)
                a.rate = sampling_rate
            return Wave(wave=a.array, sampling_rate=a.rate)
        else:
            wave, sampling_rate = librosa.core.load(
                str(path), sr=sampling_rate, dtype=dtype
            )
            return Wave(wave=wave, sampling_rate=sampling_rate)

    def save(self, path: Path):
        soundfile.write(str(path), data=self.wave, samplerate=self.sampling_rate)
