from typing import Dict

import numpy
import pyworld
from scipy.interpolate import interp1d

from acoustic_feature_extractor.data.sampling_data import SamplingData
from acoustic_feature_extractor.data.wave import Wave


class F0(SamplingData):
    r"""
    array: log f0
    """

    @staticmethod
    def form_wave(
            wave: Wave,
            frame_period: float,
            f0_floor: float,
            f0_ceil: float,
            with_vuv: bool,
    ):
        w = wave.wave.astype(numpy.float64)
        sampling_rate = wave.sampling_rate

        f0, t = pyworld.harvest(
            w,
            sampling_rate,
            frame_period=frame_period,
            f0_floor=f0_floor,
            f0_ceil=f0_ceil,
        )
        f0 = pyworld.stonemask(w, f0, t, sampling_rate)

        # voice / unvoice
        vuv = f0 != 0
        f0_log = numpy.zeros_like(f0)
        f0_log[vuv] = numpy.log(f0[vuv])

        if not with_vuv:
            array = f0_log
        else:
            f0_log_voiced = f0_log[vuv]
            t_voiced = t[vuv]

            interp = interp1d(
                t_voiced,
                f0_log_voiced,
                kind='linear',
                bounds_error=False,
                fill_value=(f0_log_voiced[0], f0_log_voiced[-1]),
            )
            f0_log = interp(t)

            array = numpy.stack([f0_log, vuv.astype(f0_log.dtype)], axis=1)

        rate = int(1000 // frame_period)
        return F0(array=array, rate=rate)

    @property
    def with_vuv(self):
        return self.array.ndim == 2 and self.array.shape[1] == 2

    def convert(
            self,
            input_statistics: Dict[str, float] = None,
            target_statistics: Dict[str, float] = None,
            input_mean: float = None,
            input_var: float = None,
            target_mean: float = None,
            target_var: float = None,
    ):
        assert (input_statistics is None) != (input_mean is None and input_var is None)
        assert (target_statistics is None) != (target_mean is None and target_var is None)

        if input_statistics is not None:
            im, iv = input_statistics['mean'], input_statistics['var']
        else:
            im, iv = input_mean, input_var

        if target_statistics is not None:
            tm, tv = target_statistics['mean'], target_statistics['var']
        else:
            tm, tv = target_mean, target_var

        if self.with_vuv:
            f0 = self.array[:, 0]
            f0 = numpy.exp((tv / iv) * (numpy.log(f0) - im) + tm)
            self.array[:, 0] = f0
        else:
            f0 = self.array
            f0[f0.nonzero()] = numpy.exp((tv / iv) * (numpy.log(f0[f0.nonzero()]) - im) + tm)
            self.array = f0
