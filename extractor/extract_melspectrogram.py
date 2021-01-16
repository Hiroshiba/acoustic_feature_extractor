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
from acoustic_feature_extractor.utility.json_utility import save_arguments


def process(
    path: Path,
    output_directory: Path,
    sampling_rate: int,
    preemph: Optional[float],
    n_mels: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    fmin: float,
    fmax: float,
    min_level: float,
    max_level: Optional[float],
    disable_normalize: bool,
):
    wave = Wave.load(path, sampling_rate)

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
        max_level=max_level,
        normalize=not disable_normalize,
    )

    rate = sampling_rate / hop_length

    out = output_directory / (path.stem + ".npy")
    numpy.save(str(out), dict(array=ms, rate=rate))


def extract_melspectrogram(
    input_glob,
    output_directory: Path,
    sampling_rate: int,
    preemph: Optional[float],
    n_mels: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    fmin: float,
    fmax: float,
    min_level: float,
    max_level: Optional[float],
    disable_normalize: bool,
):
    output_directory.mkdir(exist_ok=True)
    save_arguments(locals(), output_directory / "arguments.json")

    paths = [Path(p) for p in glob.glob(str(input_glob))]
    _process = partial(
        process,
        output_directory=output_directory,
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
        disable_normalize=disable_normalize,
    )

    with multiprocessing.Pool() as pool:
        list(
            tqdm.tqdm(
                pool.imap(_process, paths),
                total=len(paths),
                desc="extract_melspectrogram",
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glob", "-ig", required=True)
    parser.add_argument("--output_directory", "-od", type=Path, required=True)
    parser.add_argument("--sampling_rate", "-sr", type=int, default=24000)
    parser.add_argument("--preemph", type=float, default=None)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--win_length", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--fmin", type=float, default=125)
    parser.add_argument("--fmax", type=float, default=12000)
    parser.add_argument("--min_level", type=float, default=1e-5)
    parser.add_argument("--max_level", type=float)
    parser.add_argument("--disable_normalize", action="store_true")
    extract_melspectrogram(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
