import argparse
import glob
import multiprocessing
from functools import partial
from pathlib import Path

import numpy
import tqdm

from acoustic_feature_extractor.data.phoneme import PhonemeType, phoneme_type_to_class
from acoustic_feature_extractor.data.wave import Wave
from acoustic_feature_extractor.utility.json_utility import save_arguments


def process(
    paths: tuple[Path, Path],
    output_directory: Path,
    phoneme_type: PhonemeType,
    sampling_rate: int,
):
    try:
        wave_path, phoneme_path = paths

        phoneme_class = phoneme_type_to_class[phoneme_type]
        phonemes = phoneme_class.load_julius_list(phoneme_path)

        length = len(Wave.load(wave_path, sampling_rate=sampling_rate).wave)
        array = numpy.ones((length,), dtype=numpy.bool)

        for p in filter(lambda p: p.phoneme != phoneme_class.space_phoneme, phonemes):
            s = int(round(p.start * sampling_rate))
            e = int(round(p.end * sampling_rate))
            array[s : e + 1] = False

        out = output_directory / (wave_path.stem + ".npy")
        numpy.save(str(out), dict(array=array, rate=sampling_rate))

    except:
        print("error:", paths)
        raise


def extract_silence_from_phoneme(
    input_wave_glob,
    input_phoneme_glob,
    output_directory: Path,
    phoneme_type: PhonemeType,
    sampling_rate: int,
):
    output_directory.mkdir(exist_ok=True)
    save_arguments(locals(), output_directory / "arguments.json")

    wave_paths = sorted(Path(p) for p in glob.glob(input_wave_glob))
    phoneme_paths = sorted(Path(p) for p in glob.glob(input_phoneme_glob))
    assert len(wave_paths) == len(phoneme_paths)

    paths = list(zip(wave_paths, phoneme_paths, strict=False))

    _process = partial(
        process,
        output_directory=output_directory,
        phoneme_type=phoneme_type,
        sampling_rate=sampling_rate,
    )

    pool = multiprocessing.Pool()
    list(tqdm.tqdm(pool.imap(_process, paths), total=len(paths)))


def main():
    parser = argparse.ArgumentParser(
        description="音素ラベル情報から無音部分を抽出します。音素ラベルでの無音音素部分をバイナリラベルとして出力します。"
    )
    parser.add_argument(
        "--input_wave_glob",
        "-iwg",
        required=True,
        help="入力音声ファイルのパスパターン（例：'*.wav'）",
    )
    parser.add_argument(
        "--input_phoneme_glob",
        "-ipg",
        required=True,
        help="入力音素ラベルファイルのパスパターン（例：'*.lab'）",
    )
    parser.add_argument(
        "--output_directory",
        "-od",
        type=Path,
        required=True,
        help="抽出された無音ラベルを保存するディレクトリ",
    )
    parser.add_argument(
        "--phoneme_type",
        "-pt",
        type=PhonemeType,
        default=PhonemeType.seg_kit,
        help="音素の種類。seg_kit, jvs, openjtalk, rohan, kiritan, song, dummy（デフォルト：seg_kit）",
    )
    parser.add_argument(
        "--sampling_rate",
        "-sr",
        type=int,
        required=True,
        help="サンプリングレート（Hz）。出力ラベルの時間解像度を決定",
    )
    extract_silence_from_phoneme(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
