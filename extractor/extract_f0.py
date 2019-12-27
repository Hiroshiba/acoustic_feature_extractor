import argparse
import glob
import multiprocessing
from functools import partial
from pathlib import Path
from typing import Optional, Dict

import numpy
import pyworld
import tqdm
from scipy.interpolate import interp1d

from acoustic_feature_extractor.data import Wave
from acoustic_feature_extractor.utility.json_utility import save_arguments


def process(
        path: Path,
        output_directory: Path,
        sampling_rate: int,
        frame_period: float,
        f0_floor: float,
        f0_ceil: float,
        with_vuv: bool,
        input_statistics: Optional[Path],
        target_statistics: Optional[Path],
):
    w = Wave.load(path, sampling_rate).wave.astype(numpy.float64)

    f0, t = pyworld.harvest(
        w,
        sampling_rate,
        frame_period=frame_period,
        f0_floor=f0_floor,
        f0_ceil=f0_ceil,
    )
    f0 = pyworld.stonemask(w, f0, t, sampling_rate)

    if input_statistics is not None:
        stat: Dict = numpy.load(input_statistics, allow_pickle=True).item()
        im, iv = stat['mean'], stat['var']

        stat: Dict = numpy.load(target_statistics, allow_pickle=True).item()
        tm, tv = stat['mean'], stat['var']

        f0 = numpy.copy(f0)
        f0[f0.nonzero()] = numpy.exp((tv / iv) * (numpy.log(f0[f0.nonzero()]) - im) + tm)

    vuv = f0 != 0
    f0_log = numpy.zeros_like(f0)
    f0_log[vuv] = numpy.log(f0[vuv])

    if not with_vuv:
        array = f0_log
    else:
        f0_log_voiced = f0_log[vuv]
        t_voiced = t[vuv]

        interp = interp1d(
            t_voiced,
            f0_log_voiced,
            kind='linear',
            bounds_error=False,
            fill_value=(f0_log_voiced[0], f0_log_voiced[-1]),
        )
        f0_log = interp(t)

        array = numpy.stack([f0_log, vuv.astype(f0_log.dtype)], axis=1)

    rate = int(1000 // frame_period)

    out = output_directory / (path.stem + '.npy')
    numpy.save(str(out), dict(array=array, rate=rate))


def extract_f0(
        input_glob,
        output_directory: Path,
        sampling_rate: int,
        frame_period: float,
        f0_floor: float,
        f0_ceil: float,
        with_vuv: bool,
        input_statistics: Path,
        target_statistics: Path,
):
    output_directory.mkdir(exist_ok=True)
    save_arguments(locals(), output_directory / 'arguments.json')

    paths = [Path(p) for p in glob.glob(str(input_glob))]
    _process = partial(
        process,
        output_directory=output_directory,
        sampling_rate=sampling_rate,
        frame_period=frame_period,
        f0_floor=f0_floor,
        f0_ceil=f0_ceil,
        with_vuv=with_vuv,
        input_statistics=input_statistics,
        target_statistics=target_statistics,
    )

    pool = multiprocessing.Pool()
    list(tqdm.tqdm(pool.imap_unordered(_process, paths), total=len(paths)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_glob', '-ig', required=True)
    parser.add_argument('--output_directory', '-od', type=Path, required=True)
    parser.add_argument('--sampling_rate', '-sr', type=int, required=True)
    parser.add_argument('--frame_period', '-fp', type=float, default=5.0)
    parser.add_argument('--f0_floor', '-ff', type=int, default=71.0)
    parser.add_argument('--f0_ceil', '-fc', type=int, default=800.0)
    parser.add_argument('--with_vuv', '-wv', action='store_true')
    parser.add_argument('--input_statistics', '-is', type=Path)
    parser.add_argument('--target_statistics', '-ts', type=Path)
    extract_f0(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
