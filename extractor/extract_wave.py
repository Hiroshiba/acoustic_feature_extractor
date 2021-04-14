import argparse
import glob
import multiprocessing
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import numpy
import tqdm
from acoustic_feature_extractor.data.wave import Wave
from acoustic_feature_extractor.utility.json_utility import save_arguments


def process(
    path: Path,
    output_directory: Path,
    sampling_rate: int,
    clipping_range: Optional[Tuple[float, float]],
    clipping_auto: bool,
):
    w = Wave.load(path, sampling_rate).wave

    if clipping_range is not None:
        w = numpy.clip(w, clipping_range[0], clipping_range[1]) / numpy.max(
            numpy.abs(clipping_range)
        )

    if clipping_auto:
        w /= numpy.abs(w).max() * 0.999

    out = output_directory / (path.stem + ".npy")
    numpy.save(str(out), dict(array=w, rate=sampling_rate))


def extract_wave(
    input_glob,
    output_directory: Path,
    sampling_rate: int,
    clipping_range: Optional[Tuple[float, float]],
    clipping_auto: bool,
):
    assert clipping_range is None or not clipping_auto

    output_directory.mkdir(exist_ok=True)
    save_arguments(locals(), output_directory / "arguments.json")

    paths = [Path(p) for p in glob.glob(str(input_glob))]

    _process = partial(
        process,
        output_directory=output_directory,
        sampling_rate=sampling_rate,
        clipping_range=clipping_range,
        clipping_auto=clipping_auto,
    )

    pool = multiprocessing.Pool()
    list(tqdm.tqdm(pool.imap(_process, paths), total=len(paths)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glob", "-ig", required=True)
    parser.add_argument("--output_directory", "-od", type=Path, required=True)
    parser.add_argument("--sampling_rate", "-sr", type=int, required=True)
    parser.add_argument(
        "--clipping_range", "-cr", type=float, nargs=2, help="(min, max)"
    )
    parser.add_argument("--clipping_auto", "-ca", action="store_true")
    extract_wave(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
