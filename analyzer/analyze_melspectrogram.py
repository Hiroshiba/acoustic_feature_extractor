import argparse
import glob
import multiprocessing
from functools import partial
from pathlib import Path
from typing import Optional

import numpy
import tqdm
from acoustic_feature_extractor.data.spectrogram import to_log_melspectrogram
from acoustic_feature_extractor.data.wave import Wave


def process(
    path: Path,
    sampling_rate: int,
    preemph: Optional[float],
    n_mels: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    fmin: float,
    fmax: float,
    eliminate_silence: bool,
):
    wave = Wave.load(path, sampling_rate)

    min_level = 1e-15
    ms = to_log_melspectrogram(
        x=wave.wave,
        sampling_rate=sampling_rate,
        preemph=preemph,
        n_mels=n_mels,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        min_level=min_level,
        max_level=None,
        normalize=False,
    )

    if eliminate_silence:
        ms[ms == min_level] = ms[ms != min_level].min()

    return ms.min(), ms.max()


def analyze_melspectrogram(
    input_glob,
    sampling_rate: int,
    preemph: Optional[float],
    n_mels: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    fmin: float,
    fmax: float,
    eliminate_silence: bool,
):
    paths = [Path(p) for p in glob.glob(str(input_glob))]
    _process = partial(
        process,
        sampling_rate=sampling_rate,
        preemph=preemph,
        n_mels=n_mels,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        eliminate_silence=eliminate_silence,
    )

    pool = multiprocessing.Pool()
    datas = list(
        tqdm.tqdm(
            pool.imap(_process, paths), total=len(paths), desc="analyze_melspectrogram"
        )
    )

    m = numpy.array(datas)
    print("min", m[:, 0].min())
    print("max", m[:, 1].max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glob", "-ig", required=True)
    parser.add_argument("--sampling_rate", "-sr", type=int, default=24000)
    parser.add_argument("--preemph", type=float, default=None)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--win_length", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--fmin", type=float, default=125)
    parser.add_argument("--fmax", type=float, default=12000)
    parser.add_argument("--eliminate_silence", action="store_true")
    analyze_melspectrogram(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
