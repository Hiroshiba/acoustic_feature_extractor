import unittest
from typing import Optional

import librosa
import numpy
from acoustic_feature_extractor.data.spectrogram import to_log_melspectrogram


class TestToLogMelspectrogram(unittest.TestCase):
    @staticmethod
    def _to_log_melspectrogram(
        x: numpy.ndarray = None,
        sampling_rate: int = 24000,
        preemph: Optional[float] = 0.97,
        n_mels: int = 80,
        n_fft: int = 2048,
        win_length: int = 1024,
        hop_length: int = 256,
        fmin: float = 125,
        fmax: float = 12000,
        min_level: float = 1e-5,
        max_level: float = None,
        normalize: bool = True,
    ):
        if x is None:
            x, _ = librosa.load(librosa.util.example_audio_file(), sr=sampling_rate)

        return to_log_melspectrogram(
            x=x,
            sampling_rate=sampling_rate,
            preemph=preemph,
            n_mels=n_mels,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax,
            min_level=min_level,
            max_level=max_level,
            normalize=normalize,
        )

    def test_can_convert_to_log_melspectrogram(self):
        self._to_log_melspectrogram()

    def test_can_convert_to_log_melspectrogram_with_preemph(self):
        self._to_log_melspectrogram(preemph=None)
        self._to_log_melspectrogram(preemph=0.97)

    def test_can_convert_to_log_melspectrogram_with_min_level(self):
        sp = self._to_log_melspectrogram(min_level=1e-1, normalize=False)
        self.assertGreaterEqual(sp.min(), numpy.log(1e-1).astype(numpy.float32))

        sp = self._to_log_melspectrogram(min_level=1e-5, normalize=False)
        self.assertGreaterEqual(sp.min(), numpy.log(1e-5).astype(numpy.float32))

    def test_can_convert_to_log_melspectrogram_with_max_level(self):
        sp = self._to_log_melspectrogram(max_level=1e1, normalize=False)
        self.assertLessEqual(sp.max(), numpy.log(1e1))

        sp = self._to_log_melspectrogram(max_level=1e5, normalize=False)
        self.assertLessEqual(sp.max(), numpy.log(1e5))

    def test_can_convert_to_log_melspectrogram_with_normalize(self):
        sp = self._to_log_melspectrogram(min_level=1e-5, max_level=1e5, normalize=False)
        min_level = sp.min() + 1
        max_level = sp.max() - 1

        sp = self._to_log_melspectrogram(min_level=numpy.exp(min_level), normalize=True)
        self.assertEqual(sp.min(), 0)

        sp = self._to_log_melspectrogram(
            min_level=numpy.exp(min_level),
            max_level=numpy.exp(max_level),
            normalize=True,
        )
        self.assertEqual(sp.min(), 0)
        self.assertEqual(sp.max(), 1)
