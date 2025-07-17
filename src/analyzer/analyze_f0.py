import argparse
import glob
import multiprocessing
from pathlib import Path

import numpy
import tqdm

from acoustic_feature_extractor.data.f0 import F0


def load_f0(
    path: Path,
):
    f0 = F0.load(path)

    if f0.with_vuv:
        f0_log = f0.array[:, 0]
        vuv = f0.array[:, 1].astype(bool)
    else:
        f0_log = f0.array
        vuv = f0_log.nonzero()

    return numpy.exp(f0_log[vuv])


def analyze_f0(
    input_glob,
):
    paths = [Path(p) for p in sorted(glob.glob(str(input_glob)))]

    pool = multiprocessing.Pool()
    it = pool.imap(load_f0, paths)
    f0_list = list(tqdm.tqdm(it, total=len(paths), desc="load_f0"))

    f0 = numpy.concatenate(f0_list)

    q = [0, 0.1, 0.3, 0.5, 1, 99, 99.5, 99.7, 99.9, 100]
    x = numpy.round(numpy.percentile(f0, q=q)).astype(numpy.int32)

    print("q", "x")
    for q_, x_ in zip(q, x, strict=False):
        print(q_, x_)


def main():
    parser = argparse.ArgumentParser(
        description="抽出されたF0データを分析し、統計情報を表示します。特に分位数情報を表示します。"
    )
    parser.add_argument(
        "--input_glob",
        "-i",
        required=True,
        help="入力F0データファイルのパスパターン（例：'*.npy'）。extract_f0で生成されたデータを指定",
    )
    analyze_f0(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
