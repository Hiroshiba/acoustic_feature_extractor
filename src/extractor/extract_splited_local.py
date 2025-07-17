import argparse
import glob
import multiprocessing
from collections.abc import Sequence
from functools import partial
from operator import itemgetter
from pathlib import Path

import numpy
import tqdm

from acoustic_feature_extractor.data.sampling_data import SamplingData
from acoustic_feature_extractor.utility.json_utility import save_arguments


def process(
    path: Path,
    output_directory: Path,
    keypoint_seconds: numpy.ndarray,
    file_format: str,
    padding: bool,
    remove_all_same: bool,
):
    try:
        base = SamplingData.load(path)
        datas = dict(enumerate(base.split(keypoint_seconds=keypoint_seconds)))

        if remove_all_same:
            datas = {
                index: data for index, data in datas.items() if not data.all_same()
            }

        if padding:
            padding_value = base.estimate_padding_value()
            items = datas.items()
            indexes, datas = (
                list(map(itemgetter(0), items)),
                list(map(itemgetter(1), items)),
            )
            datas = SamplingData.padding(datas, padding_value=padding_value)
            datas = {key: data for key, data in zip(indexes, datas, strict=False)}

        for i, data in datas.items():
            if data is None:
                continue
            out = output_directory / file_format.format(
                stem=path.stem, suffix=path.suffix, i=i
            )
            data.save(out)

    except:
        print("error:", path)
        raise


def extract_splited_local(
    input_glob,
    output_directory: Path,
    keypoint_seconds: Sequence[float],
    file_format: str,
    padding: bool,
    remove_all_same: bool,
):
    output_directory.mkdir(exist_ok=True)
    save_arguments(locals(), output_directory / "arguments.json")

    paths = [Path(p) for p in glob.glob(str(input_glob))]
    _process = partial(
        process,
        output_directory=output_directory,
        keypoint_seconds=numpy.array(keypoint_seconds),
        file_format=file_format,
        padding=padding,
        remove_all_same=remove_all_same,
    )

    pool = multiprocessing.Pool()
    list(tqdm.tqdm(pool.imap_unordered(_process, paths), total=len(paths)))


def main():
    parser = argparse.ArgumentParser(
        description="データを指定した時刻で分割します。時系列データを特定のキーポイントで切り分けます。"
    )
    parser.add_argument(
        "--input_glob",
        "-ig",
        required=True,
        help="入力データファイルのパスパターン（例：'*.npy'）",
    )
    parser.add_argument(
        "--output_directory",
        "-od",
        type=Path,
        required=True,
        help="分割されたデータを保存するディレクトリ",
    )
    parser.add_argument(
        "--keypoint_seconds",
        "-ks",
        nargs="+",
        type=float,
        required=True,
        help="キーポイントの時刻リスト（秒）。この時刻でデータを分割します",
    )
    parser.add_argument(
        "--file_format",
        "-ff",
        default="{stem}-{i}{suffix}",
        help="出力ファイル名のフォーマット。{stem}, {suffix}, {i}を使用可能（デフォルト：{stem}-{i}{suffix}）",
    )
    parser.add_argument(
        "--padding",
        "-p",
        action="store_true",
        help="パディングを行うかどうか。指定すると分割されたデータを同じ長さに揃えます",
    )
    parser.add_argument(
        "--remove_all_same",
        "-ras",
        action="store_true",
        help="全て同じ値のデータを除去するかどうか。指定すると均一なデータをスキップします",
    )
    extract_splited_local(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
