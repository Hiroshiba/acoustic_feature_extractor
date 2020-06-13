import argparse
import glob
import multiprocessing
from functools import partial
from pathlib import Path
from typing import List

import numpy
import tqdm

from acoustic_feature_extractor.data.sampling_data import SamplingData
from acoustic_feature_extractor.utility.json_utility import save_arguments


def process(
    paths: List[Path],
    output_directory: Path,
    rate: int,
    mode: str,
    error_time_length: float,
):
    assert all(paths[0].stem == p.stem for p in paths[1:])

    datas = [SamplingData.load(p) for p in paths]
    array = SamplingData.collect(
        datas, rate=rate, mode=mode, error_time_length=error_time_length
    )

    out = output_directory / (paths[0].stem + ".npy")
    numpy.save(str(out), dict(array=array, rate=rate))


def process_ignore_error(*args, **kwargs):
    try:
        return process(*args, **kwargs)
    except Exception as e:
        return e


def extract_collected_local(
    input_glob_list,
    output_directory: Path,
    ignore_error: bool,
    rate: int,
    mode: str,
    error_time_length: float,
):
    output_directory.mkdir(exist_ok=True)
    save_arguments(locals(), output_directory / "arguments.json")

    paths_list = [
        [Path(p) for p in p_list]
        for p_list in zip(
            *[sorted(glob.glob(input_glob)) for input_glob in input_glob_list]
        )
    ]
    _process = partial(
        process if not ignore_error else process_ignore_error,
        output_directory=output_directory,
        rate=rate,
        mode=mode,
        error_time_length=error_time_length,
    )

    pool = multiprocessing.Pool()
    results = list(
        tqdm.tqdm(pool.imap_unordered(_process, paths_list), total=len(paths_list))
    )

    if ignore_error:
        errors = list(filter(None, results))
        print(f"num of error: {len(errors)}")

        for e in errors:
            print(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glob_list", "-igl", nargs="+", required=True)
    parser.add_argument("--output_directory", "-od", type=Path, required=True)
    parser.add_argument("--rate", "-r", type=int, required=True)
    parser.add_argument("--mode", "-m", choices=["min", "max"], default="min")
    parser.add_argument("--error_time_length", "-etl", type=float, default=0.015)
    parser.add_argument("--ignore_error", "-ig", action="store_true")
    extract_collected_local(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
