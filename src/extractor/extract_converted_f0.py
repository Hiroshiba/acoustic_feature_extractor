import argparse
import glob
import multiprocessing
from functools import partial
from pathlib import Path

import tqdm

from acoustic_feature_extractor.data.f0 import F0
from acoustic_feature_extractor.utility.json_utility import save_arguments
from acoustic_feature_extractor.utility.numpy_utility import load_numpy_object


def process(
    path: Path,
    output_directory: Path,
    input_statistics: Path | None,
    target_statistics: Path | None,
    input_mean: float | None,
    input_var: float | None,
    target_mean: float | None,
    target_var: float | None,
):
    try:
        f0 = F0.load(path=path).convert(
            input_statistics=(
                load_numpy_object(input_statistics)
                if input_statistics is not None
                else None
            ),
            target_statistics=(
                load_numpy_object(target_statistics)
                if target_statistics is not None
                else None
            ),
            input_mean=input_mean,
            input_var=input_var,
            target_mean=target_mean,
            target_var=target_var,
        )

        out = output_directory / (path.stem + ".npy")
        f0.save(out)

    except:
        print("error:", path)
        raise


def extract_converted_f0(
    input_glob,
    output_directory: Path,
    input_statistics: Path | None,
    target_statistics: Path | None,
    input_mean: float | None,
    input_var: float | None,
    target_mean: float | None,
    target_var: float | None,
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
    parser = argparse.ArgumentParser(
        description="F0データを統計情報を用いて正規化・変換します。入力とターゲットの統計情報（平均・分散）を使って線形変換を行います。"
    )
    parser.add_argument(
        "--input_glob",
        "-ig",
        required=True,
        help="入力F0ファイルのパスパターン（例：'*.npy'）。extract_f0で生成されたF0データを指定",
    )
    parser.add_argument(
        "--output_directory",
        "-od",
        type=Path,
        required=True,
        help="変換されたF0データを保存するディレクトリ",
    )
    parser.add_argument(
        "--input_statistics",
        "-is",
        type=Path,
        help="入力F0の統計情報ファイル（.npy形式）。extract_f0_statisticsで生成されたファイルを指定",
    )
    parser.add_argument(
        "--target_statistics",
        "-ts",
        type=Path,
        help="ターゲットF0の統計情報ファイル（.npy形式）。変換先の統計情報を指定",
    )
    parser.add_argument(
        "--input_mean",
        "-im",
        type=float,
        help="入力F0の平均値。統計情報ファイルを使わない場合に直接指定",
    )
    parser.add_argument(
        "--input_var",
        "-iv",
        type=float,
        help="入力F0の分散値。統計情報ファイルを使わない場合に直接指定",
    )
    parser.add_argument(
        "--target_mean",
        "-tm",
        type=float,
        help="ターゲットF0の平均値。統計情報ファイルを使わない場合に直接指定",
    )
    parser.add_argument(
        "--target_var",
        "-tv",
        type=float,
        help="ターゲットF0の分散値。統計情報ファイルを使わない場合に直接指定",
    )
    extract_converted_f0(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
