import argparse
import glob
import multiprocessing
from functools import partial
from pathlib import Path

import tqdm

from acoustic_feature_extractor.data.sampling_data import DegenerateType, SamplingData
from acoustic_feature_extractor.utility.json_utility import save_arguments


def process(
    path: Path,
    output_directory: Path,
    frame_length: int,
    hop_length: int,
    centering: bool,
    padding_value: int | None,
    padding_mode: str | None,
    degenerate_type: DegenerateType,
):
    SamplingData.load(path).degenerate(
        frame_length=frame_length,
        hop_length=hop_length,
        centering=centering,
        padding_value=padding_value,
        padding_mode=padding_mode,
        degenerate_type=degenerate_type,
    ).save(output_directory / (path.stem + ".npy"))


def extract_framed_local(
    input_glob,
    output_directory: Path,
    frame_length: int,
    hop_length: int,
    centering: bool,
    padding_value: int | None,
    padding_mode: str | None,
    degenerate_type: DegenerateType,
):
    output_directory.mkdir(exist_ok=True)
    save_arguments(locals(), output_directory / "arguments.json")

    paths = [Path(p) for p in glob.glob(str(input_glob))]
    _process = partial(
        process,
        output_directory=output_directory,
        frame_length=frame_length,
        hop_length=hop_length,
        centering=centering,
        padding_value=padding_value,
        padding_mode=padding_mode,
        degenerate_type=degenerate_type,
    )

    with multiprocessing.Pool() as pool:
        list(tqdm.tqdm(pool.imap_unordered(_process, paths), total=len(paths)))


def main():
    parser = argparse.ArgumentParser(
        description="データをフレーム単位で集約します。時系列データをフレームごとに統計的に集約します。"
    )
    parser.add_argument(
        "--input_glob",
        required=True,
        help="入力データファイルのパスパターン（例：'*.npy'）",
    )
    parser.add_argument(
        "--output_directory",
        type=Path,
        required=True,
        help="集約されたデータを保存するディレクトリ",
    )
    parser.add_argument(
        "--frame_length",
        type=int,
        required=True,
        help="フレーム長（サンプル数）。各フレームのサイズを指定",
    )
    parser.add_argument(
        "--hop_length",
        type=int,
        required=True,
        help="ホップ長（サンプル数）。フレーム間のステップサイズを指定",
    )
    parser.add_argument(
        "--centering",
        action="store_true",
        help="センタリングを行うかどうか。指定するとフレームの中心を基準にします",
    )
    parser.add_argument(
        "--padding_value", type=int, help="パディング値。フレームの端での補充値を指定"
    )
    parser.add_argument(
        "--padding_mode", help="パディングモード。フレームの端での補充方法を指定"
    )
    parser.add_argument(
        "--degenerate_type",
        type=DegenerateType,
        required=True,
        help="集約方法の種類。フレーム内のデータをどのように統計的に集約するかを指定（min, max, mean, median）",
    )
    extract_framed_local(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
