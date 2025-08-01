import argparse
import glob
import multiprocessing
from collections.abc import Iterable, Sequence
from functools import partial
from operator import attrgetter
from pathlib import Path

import numpy
import tqdm

from acoustic_feature_extractor.data.linguistic_feature import (
    LinguisticFeature,
    LinguisticFeatureType,
)
from acoustic_feature_extractor.data.phoneme import PhonemeType, phoneme_type_to_class
from acoustic_feature_extractor.utility.json_utility import save_arguments


def sorted_with_stem(paths: Iterable[Path]):
    return sorted(paths, key=attrgetter("stem"))


def process(
    path: Path | tuple[Path, Path, Path],
    output_directory: Path,
    phoneme_type: PhonemeType,
    rate: int,
    types: Sequence[LinguisticFeatureType],
):
    try:
        if isinstance(path, Path):
            start_accents = None
            end_accents = None
        else:
            path, start_accent_path, end_accent_path = path
            start_accents = [
                bool(int(s)) for s in start_accent_path.read_text().split()
            ]
            end_accents = [bool(int(s)) for s in end_accent_path.read_text().split()]

        phoneme_class = phoneme_type_to_class[phoneme_type]
        ps = phoneme_class.load_julius_list(path)
        array = LinguisticFeature(
            phonemes=ps,
            phoneme_class=phoneme_class,
            rate=rate,
            feature_types=types,
            start_accents=start_accents,
            end_accents=end_accents,
        ).make_array()

        out = output_directory / (path.stem + ".npy")
        numpy.save(str(out), dict(array=array, rate=rate))

    except:
        print("error:", path)
        raise


def process_ignore_error(*args, **kwargs):
    try:
        return process(*args, **kwargs)
    except Exception as e:
        return e


def extract_phoneme(
    input_glob: str,
    output_directory: Path,
    input_start_accent_glob: str | None,
    input_end_accent_glob: str | None,
    phoneme_type: PhonemeType,
    with_phoneme_id: bool,
    with_pre_post: bool,
    with_duration: bool,
    with_relative_pos: bool,
    rate: int,
    ignore_error: bool,
):
    output_directory.mkdir(exist_ok=True)
    save_arguments(locals(), output_directory / "arguments.json")

    # Linguistic Feature Type
    with_accent = input_start_accent_glob is not None

    types = []

    if not with_phoneme_id:
        types += [LinguisticFeatureType.PHONEME]
    else:
        types += [LinguisticFeatureType.PHONEME_ID]

    if with_pre_post:
        types += [
            LinguisticFeatureType.PRE_PHONEME,
            LinguisticFeatureType.POST_PHONEME,
        ]

    if with_duration:
        types += [LinguisticFeatureType.PHONEME_DURATION]

        if with_pre_post:
            types += [
                LinguisticFeatureType.PRE_PHONEME_DURATION,
                LinguisticFeatureType.POST_PHONEME_DURATION,
            ]

    if with_relative_pos:
        types += [LinguisticFeatureType.POS_IN_PHONEME]

    if with_accent:
        types += [LinguisticFeatureType.ACCENT]

    print("types:", [t.value for t in types])

    paths = [Path(p) for p in glob.glob(str(input_glob))]

    if with_accent:
        paths = sorted_with_stem(paths)
        start_accent_paths = sorted_with_stem(
            [Path(p) for p in glob.glob(str(input_start_accent_glob))]
        )
        end_accent_paths = sorted_with_stem(
            [Path(p) for p in glob.glob(str(input_end_accent_glob))]
        )

        paths = [
            (p1, p2, p3)
            for p1, p2, p3 in zip(
                paths, start_accent_paths, end_accent_paths, strict=False
            )
        ]

    _process = partial(
        process if not ignore_error else process_ignore_error,
        output_directory=output_directory,
        phoneme_type=phoneme_type,
        rate=rate,
        types=types,
    )

    with multiprocessing.Pool() as pool:
        results = list(
            tqdm.tqdm(pool.imap_unordered(_process, paths), total=len(paths))
        )

    if ignore_error:
        errors = list(filter(None, results))
        print(f"num of error: {len(errors)}")

        for e in errors:
            print(e)


def main():
    parser = argparse.ArgumentParser(
        description="音素情報ファイルから言語的特徴量を抽出します。アクセント情報、継続時間、位置情報等の特徴量を含められます。"
    )
    parser.add_argument(
        "--input_glob",
        "-ig",
        required=True,
        help="入力音素ファイルのパスパターン（例：'*.txt'）。Julius形式の音素ファイルを指定",
    )
    parser.add_argument(
        "--output_directory",
        "-od",
        type=Path,
        required=True,
        help="抽出された音素特徴量を保存するディレクトリ",
    )
    parser.add_argument(
        "--phoneme_type",
        "-pt",
        type=PhonemeType,
        default=PhonemeType.seg_kit,
        help="音素の種類。seg_kit, jvs, openjtalk, rohan, kiritan, song, dummy（デフォルト：seg_kit）",
    )
    parser.add_argument(
        "--input_start_accent_glob",
        "-isag",
        help="開始アクセント情報ファイルのパスパターン。指定すると音素にアクセント情報が追加されます",
    )
    parser.add_argument(
        "--input_end_accent_glob",
        "-ieag",
        help="終了アクセント情報ファイルのパスパターン。開始アクセントと合わせて指定",
    )
    parser.add_argument(
        "--with_phoneme_id",
        "-wpi",
        action="store_true",
        help="音素IDを含めるかどうか。指定すると音素を数値IDで表現します",
    )
    parser.add_argument(
        "--with_pre_post",
        "-wpp",
        action="store_true",
        help="前後の音素情報を含めるかどうか。指定すると前音素・後音素の情報も出力されます",
    )
    parser.add_argument(
        "--with_duration",
        "-wd",
        action="store_true",
        help="継続時間情報を含めるかどうか。指定すると音素の継続時間情報が追加されます",
    )
    parser.add_argument(
        "--with_relative_pos",
        "-wrp",
        action="store_true",
        help="相対位置情報を含めるかどうか。指定すると音素内の相対位置情報が追加されます",
    )
    parser.add_argument(
        "--rate",
        "-r",
        type=int,
        default=100,
        help="出力データのサンプリングレート（Hz）。特徴量の時間解像度を決定（デフォルト：100Hz）",
    )
    parser.add_argument(
        "--ignore_error",
        "-ie",
        action="store_true",
        help="エラーを無視するかどうか。指定すると処理エラーが発生してもスキップして続行します",
    )
    extract_phoneme(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
