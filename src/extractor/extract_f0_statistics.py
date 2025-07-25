import argparse
import glob
import multiprocessing
from functools import partial
from pathlib import Path

import numpy
import tqdm

from acoustic_feature_extractor.data.f0 import F0


def load_f0_log(
    path: Path,
):
    return F0.load(path).valid_f0_log


def extract_f0_statistics(
    input_glob,
    output: Path,
):
    paths = [Path(p) for p in sorted(glob.glob(str(input_glob)))]

    _process = partial(
        load_f0_log,
    )

    pool = multiprocessing.Pool()
    it = pool.imap(_process, paths)
    f0_log_list = list(tqdm.tqdm(it, total=len(paths), desc="load_f0_log"))

    f0_log = numpy.concatenate(f0_log_list)

    mean, var = f0_log.mean(), f0_log.var()
    numpy.save(output, dict(mean=mean, var=var))


def main():
    parser = argparse.ArgumentParser(
        description="F0データファイルから統計情報（平均値・分散）を計算します。F0正規化やピッチ変換の前処理に使用されます。"
    )
    parser.add_argument(
        "--input_glob",
        "-i",
        required=True,
        help="入力F0ファイルのパスパターン（例：'*.npy'）。extract_f0で生成されたF0データを指定",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="統計情報を保存するファイルパス（.npy形式）",
    )
    extract_f0_statistics(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
