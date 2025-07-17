import argparse
import glob
import multiprocessing
from functools import partial
from pathlib import Path

import tqdm

from acoustic_feature_extractor.data.f0 import F0, F0Type
from acoustic_feature_extractor.data.wave import Wave
from acoustic_feature_extractor.utility.json_utility import save_arguments


def process(
    path: Path,
    output_directory: Path,
    sampling_rate: int | None,
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
    parser = argparse.ArgumentParser(
        description="音声ファイルから基本周波数(F0)を抽出します。"
    )
    parser.add_argument(
        "--input_glob",
        "-ig",
        required=True,
        help="入力音声ファイルのパスパターン（例：'*.wav'）",
    )
    parser.add_argument(
        "--output_directory",
        "-od",
        type=Path,
        required=True,
        help="抽出されたF0データを保存するディレクトリ",
    )
    parser.add_argument(
        "--sampling_rate",
        "-sr",
        type=int,
        help="サンプリングレート（Hz）。指定しない場合は元ファイルのサンプリングレートを使用",
    )
    parser.add_argument(
        "--frame_period",
        "-fp",
        type=float,
        default=5.0,
        help="フレーム周期（ミリ秒）。F0分析のフレーム間隔を指定（デフォルト：5.0ms）",
    )
    parser.add_argument(
        "--f0_floor",
        "-ff",
        type=float,
        default=71.0,
        help="F0分析の最低周波数（Hz）。これより低い周波数は無視される（デフォルト：71.0Hz）",
    )
    parser.add_argument(
        "--f0_ceil",
        "-fc",
        type=float,
        default=800.0,
        help="F0分析の最高周波数（Hz）。これより高い周波数は無視される（デフォルト：800.0Hz）",
    )
    parser.add_argument(
        "--with_vuv",
        "-wv",
        action="store_true",
        help="有声無声情報（VUV）を含めるかどうか。指定すると出力に有声無声フラグが追加されます",
    )
    parser.add_argument(
        "--f0_type",
        "-ft",
        type=F0Type,
        default=F0Type.world,
        help="F0抽出アルゴリズムの種類。true_world: Harvestアルゴリズムのみ、world: Harvest+StoneMask精密化、refine_world: Harvest+非周期音除去（デフォルト：world）",
    )
    extract_f0(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
