import argparse
import glob
import multiprocessing
from functools import partial
from pathlib import Path
from typing import Optional

import tqdm
from acoustic_feature_extractor.data.f0 import F0, F0Type
from acoustic_feature_extractor.data.wave import Wave
from acoustic_feature_extractor.utility.json_utility import save_arguments


def process(
    path: Path,
    output_directory: Path,
    sampling_rate: Optional[int],
    frame_period: float,
    f0_floor: float,
    f0_ceil: float,
    with_vuv: bool,
    f0_type: F0Type,
):
    f0 = F0.from_wave(
        wave=Wave.load(path, sampling_rate),
        frame_period=frame_period,
        f0_floor=f0_floor,
        f0_ceil=f0_ceil,
        with_vuv=with_vuv,
        f0_type=f0_type,
    )

    out = output_directory / (path.stem + ".npy")
    f0.save(out)


def extract_f0(
    input_glob,
    output_directory: Path,
    sampling_rate: int,
    frame_period: float,
    f0_floor: float,
    f0_ceil: float,
    with_vuv: bool,
    f0_type: F0Type,
):
    output_directory.mkdir(exist_ok=True)
    save_arguments(locals(), output_directory / "arguments.json")

    paths = [Path(p) for p in glob.glob(str(input_glob))]
    _process = partial(
        process,
        output_directory=output_directory,
        sampling_rate=sampling_rate,
        frame_period=frame_period,
        f0_floor=f0_floor,
        f0_ceil=f0_ceil,
        with_vuv=with_vuv,
        f0_type=f0_type,
    )

    with multiprocessing.Pool() as pool:
        list(tqdm.tqdm(pool.imap_unordered(_process, paths), total=len(paths)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glob", "-ig", required=True)
    parser.add_argument("--output_directory", "-od", type=Path, required=True)
    parser.add_argument("--sampling_rate", "-sr", type=int)
    parser.add_argument("--frame_period", "-fp", type=float, default=5.0)
    parser.add_argument("--f0_floor", "-ff", type=int, default=71.0)
    parser.add_argument("--f0_ceil", "-fc", type=int, default=800.0)
    parser.add_argument("--with_vuv", "-wv", action="store_true")
    parser.add_argument("--f0_type", "-ft", type=F0Type, default=F0Type.world)
    extract_f0(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
