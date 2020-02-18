import argparse
import glob
import multiprocessing
from functools import partial
from pathlib import Path

import numpy
import tqdm

from acoustic_feature_extractor.data.sampling_data import SamplingData


def load_f0_log(
        path: Path,
        with_vuv: bool,
):
    array = SamplingData.load(path).array

    if with_vuv:
        assert array.shape[1] == 2
        f0_log = array[:, 0]
        vuv = array[:, 1].astype(bool)
    else:
        assert array.ndim == 1 or array.shape[1] == 1
        f0_log = array
        vuv = f0_log.nonzero()

    return f0_log[vuv]


def extract_f0_statistics(
        input_glob,
        output: Path,
        with_vuv: bool,
):
    paths = [Path(p) for p in sorted(glob.glob(str(input_glob)))]

    _process = partial(
        load_f0_log,
        with_vuv=with_vuv,
    )

    pool = multiprocessing.Pool()
    it = pool.imap(_process, paths)
    f0_log_list = list(tqdm.tqdm(it, total=len(paths), desc='load_f0_log'))

    f0_log = numpy.concatenate(f0_log_list)

    mean, var = f0_log.mean(), f0_log.var()
    numpy.save(output, dict(mean=mean, var=var))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_glob', '-i', required=True)
    parser.add_argument('--output', '-o', type=Path, required=True)
    parser.add_argument('--with_vuv', '-wv', action='store_true')
    extract_f0_statistics(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
