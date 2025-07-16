import argparse
import glob
import multiprocessing
from functools import partial
from pathlib import Path

import numpy
import tqdm

from acoustic_feature_extractor.data.sampling_data import SamplingData
from acoustic_feature_extractor.utility.json_utility import save_arguments


def process(
    path: Path,
    output_directory: Path,
    sampling_rate: float | None,
    frame_second: float | None,
    time_axis: int,
):
    try:
        assert (sampling_rate is None) != (frame_second is None)

        if frame_second is not None:
            sampling_rate = 1 / frame_second

        assert sampling_rate is not None

        array = numpy.load(path)
        if time_axis != 0:
            array = numpy.moveaxis(array, time_axis, 0)

        data = SamplingData(array=array, rate=sampling_rate)

        out = output_directory / (path.stem + ".npy")
        data.save(out)

    except:
        print("error:", path)
        raise


def extract_sampling_data(
    input_glob,
    output_directory: Path,
    sampling_rate: float | None,
    frame_second: float | None,
    time_axis: int,
):
    output_directory.mkdir(exist_ok=True)
    save_arguments(locals(), output_directory / "arguments.json")

    paths = [Path(p) for p in glob.glob(str(input_glob))]
    _process = partial(
        process,
        output_directory=output_directory,
        sampling_rate=sampling_rate,
        frame_second=frame_second,
        time_axis=time_axis,
    )

    with multiprocessing.Pool() as pool:
        list(tqdm.tqdm(pool.imap_unordered(_process, paths), total=len(paths)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glob", "-ig", required=True)
    parser.add_argument("--output_directory", "-od", type=Path, required=True)
    parser.add_argument("--sampling_rate", "-sr", type=float)
    parser.add_argument("--frame_second", "-fs", type=float)
    parser.add_argument("--time_axis", "-ta", type=int, default=0)
    extract_sampling_data(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
