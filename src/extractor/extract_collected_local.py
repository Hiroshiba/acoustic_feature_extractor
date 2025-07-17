import argparse
import glob
import multiprocessing
from functools import partial
from pathlib import Path

import numpy
import tqdm

from acoustic_feature_extractor.data.sampling_data import SamplingData
from acoustic_feature_extractor.utility.json_utility import save_arguments


def process(
    paths: list[Path],
    output_directory: Path,
    rate: int,
    mode: str,
    error_time_length: float,
):
    try:
        assert all(paths[0].stem == p.stem for p in paths[1:])

        datas = [SamplingData.load(p) for p in paths]
        array = SamplingData.collect(
            datas, rate=rate, mode=mode, error_time_length=error_time_length
        )

        out = output_directory / (paths[0].stem + ".npy")
        numpy.save(str(out), dict(array=array, rate=rate))
    except:
        print("error:", paths)
        raise


def process_ignore_error(*args, **kwargs):
    try:
        return process(*args, **kwargs)
    except Exception as e:
        return e


def extract_collected_local(
    input_glob_list,
    output_directory: Path,
    ignore_error: bool,
    rate: int,
    mode: str,
    error_time_length: float,
    only_union: bool,
):
    output_directory.mkdir(exist_ok=True)
    save_arguments(locals(), output_directory / "arguments.json")

    paths_list = [
        {Path(p).stem: Path(p) for p in glob.glob(input_glob)}
        for input_glob in input_glob_list
    ]

    names_list = [set(n for n in paths.keys()) for paths in paths_list]

    names = names_list[0]
    if not only_union:
        for i in range(1, len(names_list)):
            assert names == names_list[i]
    else:
        for i in range(1, len(names_list)):
            names = names & names_list[i]

    pair_paths_list = [[paths[name] for paths in paths_list] for name in names]

    _process = partial(
        process if not ignore_error else process_ignore_error,
        output_directory=output_directory,
        rate=rate,
        mode=mode,
        error_time_length=error_time_length,
    )

    with multiprocessing.Pool() as pool:
        results = list(
            tqdm.tqdm(
                pool.imap_unordered(_process, pair_paths_list),
                total=len(pair_paths_list),
            )
        )

    if ignore_error:
        errors = list(filter(None, results))
        print(f"num of error: {len(errors)}")

        for e in errors:
            print(e)


def main():
    parser = argparse.ArgumentParser(
        description="複数のローカルデータを結合します。异なるソースからのデータを時間同期して結合します。"
    )
    parser.add_argument(
        "--input_glob_list",
        "-igl",
        nargs="+",
        required=True,
        help="入力データファイルのパスパターンリスト。複数のパターンを指定して結合します",
    )
    parser.add_argument(
        "--output_directory",
        "-od",
        type=Path,
        required=True,
        help="結合されたデータを保存するディレクトリ",
    )
    parser.add_argument(
        "--rate",
        "-r",
        type=int,
        required=True,
        help="出力データのサンプリングレート（Hz）",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["min", "max", "first"],
        default="min",
        help="データ結合時のモード。min: 最小値、max: 最大値、first: 最初の値（デフォルト：min）",
    )
    parser.add_argument(
        "--error_time_length",
        "-etl",
        type=float,
        default=0.015,
        help="許容エラー時間長（秒）。この範囲内の時間ズレは許容されます（デフォルト：0.015秒）",
    )
    parser.add_argument(
        "--ignore_error",
        "-ig",
        action="store_true",
        help="エラーを無視するかどうか。指定すると処理エラーが発生してもスキップして続行します",
    )
    parser.add_argument(
        "--only_union",
        "-uo",
        action="store_true",
        help="共通ファイルのみ処理するかどうか。指定すると全てのソースに存在するファイルのみ処理します",
    )
    extract_collected_local(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
