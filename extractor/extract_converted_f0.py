import argparse
import glob
import multiprocessing
from functools import partial
from pathlib import Path
from typing import Optional

import tqdm

from acoustic_feature_extractor.data.f0 import F0
from acoustic_feature_extractor.utility.json_utility import save_arguments
from acoustic_feature_extractor.utility.numpy_utility import load_numpy_object


def process(
    path: Path,
    output_directory: Path,
    input_statistics: Optional[Path],
    target_statistics: Optional[Path],
    input_mean: Optional[float],
    input_var: Optional[float],
    target_mean: Optional[float],
    target_var: Optional[float],
):
    f0 = F0.load(path=path).convert(
        input_statistics=load_numpy_object(input_statistics)
        if input_statistics is not None
        else None,
        target_statistics=load_numpy_object(target_statistics)
        if target_statistics is not None
        else None,
        input_mean=input_mean,
        input_var=input_var,
        target_mean=target_mean,
        target_var=target_var,
    )

    out = output_directory / (path.stem + ".npy")
    f0.save(out)


def extract_converted_f0(
    input_glob,
    output_directory: Path,
    input_statistics: Optional[Path],
    target_statistics: Optional[Path],
    input_mean: Optional[float],
    input_var: Optional[float],
    target_mean: Optional[float],
    target_var: Optional[float],
):
    assert (input_statistics is None) != (input_mean is None and input_var is None)
    assert (target_statistics is None) != (target_mean is None and target_var is None)

    output_directory.mkdir(exist_ok=True)
    save_arguments(locals(), output_directory / "arguments.json")

    paths = [Path(p) for p in glob.glob(str(input_glob))]
    _process = partial(
        process,
        output_directory=output_directory,
        input_statistics=input_statistics,
        target_statistics=target_statistics,
        input_mean=input_mean,
        input_var=input_var,
        target_mean=target_mean,
        target_var=target_var,
    )

    pool = multiprocessing.Pool()
    list(tqdm.tqdm(pool.imap_unordered(_process, paths), total=len(paths)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glob", "-ig", required=True)
    parser.add_argument("--output_directory", "-od", type=Path, required=True)
    parser.add_argument("--input_statistics", "-is", type=Path)
    parser.add_argument("--target_statistics", "-ts", type=Path)
    parser.add_argument("--input_mean", "-im", type=float)
    parser.add_argument("--input_var", "-iv", type=float)
    parser.add_argument("--target_mean", "-tm", type=float)
    parser.add_argument("--target_var", "-tv", type=float)
    extract_converted_f0(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
