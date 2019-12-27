from typing import Optional

import librosa
import numpy
import numpy
import pysptk
import scipy.signal


def to_log_melspectrogram(
        x: numpy.ndarray,
        sampling_rate: int,
        preemph: Optional[float],
        n_mels: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        fmin: float,
        fmax: float,
        min_level_db: float,
        normalize: bool,
):
    # pre emphasis
    if preemph is not None:
        x = scipy.signal.lfilter([1, -preemph], [1], x)

    # to mel spectrogram
    sp = numpy.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
    mel_basis = librosa.filters.mel(sampling_rate, n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    sp = numpy.dot(mel_basis, sp)

    # to log scale
    min_level = 10 ** (min_level_db / 20)
    sp = 20 * numpy.log10(numpy.maximum(min_level, sp))

    # normalize
    if normalize:
        sp = numpy.clip((sp - min_level_db) / -min_level_db, 0, 1)

    return sp.astype(numpy.float32).T[:-1]


def to_melcepstrum(
        x: numpy.ndarray,
        sampling_rate: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        order: int,
):
    sp = numpy.abs(librosa.stft(y=x, n_fft=n_fft, win_length=win_length, hop_length=hop_length)) ** 2
    sp = sp.T

    sp[sp < 1e-5] = 1e-5
    mc = pysptk.sp2mc(sp, order=order, alpha=pysptk.util.mcepalpha(sampling_rate))
    return mc
