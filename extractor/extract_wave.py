import argparse
import glob
import multiprocessing
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy
import tqdm
from acoustic_feature_extractor.data.wave import Wave
from acoustic_feature_extractor.utility.json_utility import save_arguments


def process(
    path: Path,
    output_directory: Path,
    sampling_rate: int,
    scale_db: Optional[float],
    clipping_range: Optional[Tuple[float, float]],
    clipping_auto: bool,
):
    try:
        assert (
            sum((scale_db is not None, clipping_range is not None, clipping_auto)) <= 1
        )

        w = Wave.load(path, sampling_rate).wave

        if scale_db is not None:
            w *= librosa.db_to_amplitude(scale_db)

        elif clipping_range is not None:
            w = numpy.clip(w, clipping_range[0], clipping_range[1]) / numpy.max(
                numpy.abs(clipping_range)
            )

        elif clipping_auto:
            w /= numpy.abs(w).max() * 0.999

        out = output_directory / (path.stem + ".npy")
        numpy.save(str(out), dict(array=w, rate=sampling_rate))

    except:
        print("error:", path)
        raise


def extract_wave(
    input_glob,
    output_directory: Path,
    sampling_rate: int,
    scale_db: Optional[float],
    clipping_range: Optional[Tuple[float, float]],
    clipping_auto: bool,
):
    assert sum((scale_db is not None, clipping_range is not None, clipping_auto)) <= 1

    output_directory.mkdir(exist_ok=True)
    save_arguments(locals(), output_directory / "arguments.json")

    paths = [Path(p) for p in glob.glob(str(input_glob))]

    _process = partial(
        process,
        output_directory=output_directory,
        sampling_rate=sampling_rate,
        scale_db=scale_db,
        clipping_range=clipping_range,
        clipping_auto=clipping_auto,
    )

    with multiprocessing.Pool() as pool:
        list(tqdm.tqdm(pool.imap(_process, paths), total=len(paths)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glob", "-ig", required=True)
    parser.add_argument("--output_directory", "-od", type=Path, required=True)
    parser.add_argument("--sampling_rate", "-sr", type=int, required=True)
    parser.add_argument("--scale_db", "-sd", type=float)
    parser.add_argument(
        "--clipping_range", "-cr", type=float, nargs=2, help="(min, max)"
    )
    parser.add_argument("--clipping_auto", "-ca", action="store_true")
    extract_wave(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
