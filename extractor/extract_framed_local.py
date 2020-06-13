import argparse
import glob
import multiprocessing
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Optional

import librosa
import numpy
import tqdm

from acoustic_feature_extractor.data.sampling_data import SamplingData
from acoustic_feature_extractor.utility.json_utility import save_arguments


class DegenerateType(str, Enum):
    min = "min"
    max = "max"
    mean = "mean"
    median = "median"


def process(
    path: Path,
    output_directory: Path,
    frame_length: int,
    hop_length: int,
    centering: bool,
    padding_value: Optional[int],
    padding_mode: Optional[str],
    degenerate_type: DegenerateType,
):
    data = SamplingData.load(path)
    array = data.array

    if centering:
        width = [[frame_length // 2, frame_length // 2]] + [[0, 0]] * (array.ndim - 1)
        array = numpy.pad(
            array, width, mode=padding_mode, constant_values=padding_value
        )

    array = numpy.ascontiguousarray(array)
    frame = librosa.util.frame(
        array, frame_length=frame_length, hop_length=hop_length, axis=0
    )

    if degenerate_type == DegenerateType.min:
        array = frame.min(axis=1)
    elif degenerate_type == DegenerateType.max:
        array = frame.max(axis=1)
    elif degenerate_type == DegenerateType.mean:
        array = frame.mean(axis=1)
    elif degenerate_type == DegenerateType.median:
        array = frame.median(axis=1)
    else:
        raise ValueError(degenerate_type)

    out = output_directory / (path.stem + ".npy")
    numpy.save(str(out), dict(array=array, rate=data.rate / hop_length))


def extract_framed_local(
    input_glob,
    output_directory: Path,
    frame_length: int,
    hop_length: int,
    centering: bool,
    padding_value: Optional[int],
    padding_mode: Optional[str],
    degenerate_type: DegenerateType,
):
    output_directory.mkdir(exist_ok=True)
    save_arguments(locals(), output_directory / "arguments.json")

    paths = [Path(p) for p in glob.glob(str(input_glob))]
    _process = partial(
        process,
        output_directory=output_directory,
        frame_length=frame_length,
        hop_length=hop_length,
        centering=centering,
        padding_value=padding_value,
        padding_mode=padding_mode,
        degenerate_type=degenerate_type,
    )

    with multiprocessing.Pool() as pool:
        list(tqdm.tqdm(pool.imap_unordered(_process, paths), total=len(paths)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glob", required=True)
    parser.add_argument("--output_directory", type=Path, required=True)
    parser.add_argument("--frame_length", type=int, required=True)
    parser.add_argument("--hop_length", type=int, required=True)
    parser.add_argument("--centering", action="store_true")
    parser.add_argument("--padding_value", type=int)
    parser.add_argument("--padding_mode")
    parser.add_argument("--degenerate_type", type=DegenerateType, required=True)
    extract_framed_local(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
