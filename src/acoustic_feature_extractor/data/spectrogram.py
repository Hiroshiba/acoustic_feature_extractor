import librosa
import numpy
import scipy.signal


def to_log_melspectrogram(
    x: numpy.ndarray,
    sampling_rate: int,
    preemph: float | None,
    n_mels: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    fmin: float,
    fmax: float,
    min_level: float,
    max_level: float | None,
    normalize: bool,
):
    # pre emphasis
    if preemph is not None:
        x = scipy.signal.lfilter([1, -preemph], [1], x)

    # to mel spectrogram
    sp = numpy.abs(
        librosa.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    )
    mel_basis = librosa.filters.mel(
        sr=sampling_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
    )
    sp = numpy.dot(mel_basis, sp)

    # to log scale
    log_sp = numpy.log(sp)

    log_min = numpy.log(min_level)
    log_sp = numpy.maximum(log_min, log_sp)

    if max_level is not None:
        log_sp = numpy.minimum(numpy.log(max_level), log_sp)

    # normalize
    if normalize:
        if max_level is None:
            log_sp = numpy.clip((log_sp - log_min) / -log_min, 0, 1)
        else:
            log_sp = (log_sp - log_min) / (numpy.log(max_level) - log_min)

    return log_sp.astype(numpy.float32).T
