import argparse
import glob
import json
import multiprocessing
from functools import partial
from pathlib import Path

import numpy
import tqdm

from acoustic_feature_extractor.data.wave import Wave
from acoustic_feature_extractor.process.ffmpeg import Ebur128, calc_ebur128


def process(path: Path, sampling_rate: int):
    wave = Wave.load(path, sampling_rate)
    return calc_ebur128(wave)


def analyze_ebur128(input_glob: str, sampling_rate: int):
    paths = [Path(p) for p in glob.glob(input_glob)]
    assert len(paths) > 0, "No files found"

    _process = partial(process, sampling_rate=sampling_rate)

    with multiprocessing.Pool() as pool:
        all_data: list[Ebur128] = list(
            tqdm.tqdm(pool.imap(_process, paths), total=len(paths))
        )

    all_I = numpy.array([d.I for d in all_data])
    all_LRA_high = numpy.array([d.LRA_high for d in all_data])
    all_LRA_low = numpy.array([d.LRA_low for d in all_data])

    stats = json.dumps(
        {
            "I": {
                "mean": all_I.mean(),
                "median": numpy.median(all_I),
                "max": all_I.max(),
                "min": all_I.min(),
            },
            "LRA_high": {
                "mean": all_LRA_high.mean(),
                "median": numpy.median(all_LRA_high),
                "max": all_LRA_high.max(),
                "min": all_LRA_high.min(),
            },
            "LRA_low": {
                "mean": all_LRA_low.mean(),
                "median": numpy.median(all_LRA_low),
                "max": all_LRA_low.max(),
                "min": all_LRA_low.min(),
            },
        }
    )
    print(stats)


def main():
    parser = argparse.ArgumentParser(
        description="音声ファイルのEBU R128準拠のラウドネスを分析し、統計情報を表示します。放送標準のラウドネス測定を行います。"
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
    analyze_ebur128(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
