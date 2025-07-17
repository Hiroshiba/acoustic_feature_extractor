import argparse
import glob
import multiprocessing
from functools import partial
from pathlib import Path

import librosa
import numpy
import tqdm

from acoustic_feature_extractor.data.wave import Wave
from acoustic_feature_extractor.utility.json_utility import save_arguments


def process(p: Path, sampling_rate: int):
    return Wave.load(p, sampling_rate).wave


def extract_silence(
    input_glob,
    output_directory: Path,
    sampling_rate: int,
    silence_top_db: float,
):
    output_directory.mkdir(exist_ok=True)
    save_arguments(locals(), output_directory / "arguments.json")

    paths = [Path(p) for p in glob.glob(str(input_glob))]

    _process = partial(process, sampling_rate=sampling_rate)

    with multiprocessing.Pool() as pool:
        waves = list(tqdm.tqdm(pool.imap(_process, paths), total=len(paths)))
    lengths = [len(w) for w in waves]

    wave = numpy.concatenate(waves)
    intervals = librosa.effects.split(wave, top_db=silence_top_db)
    silence = numpy.ones(len(wave), dtype=bool)

    for s, t in intervals:
        silence[s:t] = False

    for i, (s, l) in enumerate(zip(numpy.cumsum([0] + lengths), lengths, strict=False)):  # noqa: E741
        out = output_directory / (paths[i].stem + ".npy")
        numpy.save(str(out), dict(array=silence[s : s + l], rate=sampling_rate))


def main():
    parser = argparse.ArgumentParser(
        description="音声ファイルから無音部分を検出・抽出します。音量レベルに基づいて無音区間を判定し、バイナリラベルを生成します。"
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
        help="抽出された無音ラベルを保存するディレクトリ",
    )
    parser.add_argument(
        "--sampling_rate",
        "-sr",
        type=int,
        required=True,
        help="サンプリングレート（Hz）。出力ラベルの時間解像度を決定",
    )
    parser.add_argument(
        "--silence_top_db",
        "-st",
        type=float,
        default=60,
        help="無音判定の閾値（dB）。この値より小さい音量を無音とする（デフォルト：60dB）",
    )
    extract_silence(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
