import argparse
import glob
import multiprocessing
from functools import partial
from pathlib import Path
from typing import NamedTuple

import tqdm

from acoustic_feature_extractor.data.wave import Wave


class WaveData(NamedTuple):
    max: float
    min: float


def process(
    path: Path,
    sampling_rate: int,
):
    w = Wave.load(path, sampling_rate).wave
    return WaveData(
        max=w.max(),
        min=w.min(),
    )


def analyze_wave(
    input_glob,
    sampling_rate: int,
):
    paths = [Path(p) for p in glob.glob(str(input_glob))]

    _process = partial(
        process,
        sampling_rate=sampling_rate,
    )

    pool = multiprocessing.Pool()
    all_data = list(tqdm.tqdm(pool.imap(_process, paths), total=len(paths)))

    print("max", max([d.max for d in all_data]))
    print("min", min([d.min for d in all_data]))


def main():
    parser = argparse.ArgumentParser(
        description="音声ファイルの波形データを分析し、最大値・最小値を表示します。正規化パラメータの決定に使用します。"
    )
    parser.add_argument(
        "--input_glob",
        "-ig",
        required=True,
        help="入力音声ファイルのパスパターン（例：'*.wav'）",
    )
    parser.add_argument(
        "--sampling_rate",
        "-sr",
        type=int,
        required=True,
        help="サンプリングレート（Hz）",
    )
    analyze_wave(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
