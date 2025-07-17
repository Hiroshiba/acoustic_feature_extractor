import argparse
import glob
import multiprocessing
from functools import partial
from pathlib import Path

import numpy
import tqdm

from acoustic_feature_extractor.data.spectrogram import to_log_melspectrogram
from acoustic_feature_extractor.data.wave import Wave
from acoustic_feature_extractor.utility.json_utility import save_arguments


def process(
    path: Path,
    output_directory: Path,
    sampling_rate: int,
    preemph: float | None,
    n_mels: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    fmin: float,
    fmax: float,
    min_level: float,
    max_level: float | None,
    disable_normalize: bool,
):
    wave = Wave.load(path, sampling_rate)

    ms = to_log_melspectrogram(
        x=wave.wave,
        sampling_rate=sampling_rate,
        preemph=preemph,
        n_mels=n_mels,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        min_level=min_level,
        max_level=max_level,
        normalize=not disable_normalize,
    )

    rate = sampling_rate / hop_length

    out = output_directory / (path.stem + ".npy")
    numpy.save(str(out), dict(array=ms, rate=rate))


def extract_melspectrogram(
    input_glob,
    output_directory: Path,
    sampling_rate: int,
    preemph: float | None,
    n_mels: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    fmin: float,
    fmax: float,
    min_level: float,
    max_level: float | None,
    disable_normalize: bool,
):
    output_directory.mkdir(exist_ok=True)
    save_arguments(locals(), output_directory / "arguments.json")

    paths = [Path(p) for p in glob.glob(str(input_glob))]
    _process = partial(
        process,
        output_directory=output_directory,
        sampling_rate=sampling_rate,
        preemph=preemph,
        n_mels=n_mels,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        min_level=min_level,
        max_level=max_level,
        disable_normalize=disable_normalize,
    )

    with multiprocessing.Pool() as pool:
        list(
            tqdm.tqdm(
                pool.imap(_process, paths),
                total=len(paths),
                desc="extract_melspectrogram",
            )
        )


def main():
    parser = argparse.ArgumentParser(
        description="音声ファイルからメルスペクトログラムを抽出します。音声の周波数特性を可視化・分析するための特徴量です。"
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
        help="抽出されたメルスペクトログラムを保存するディレクトリ",
    )
    parser.add_argument(
        "--sampling_rate",
        "-sr",
        type=int,
        default=24000,
        help="サンプリングレート（Hz）（デフォルト：24000）",
    )
    parser.add_argument(
        "--preemph",
        type=float,
        default=None,
        help="プリエンファシス係数。高周波成分を強調する係数（デフォルト：なし）",
    )
    parser.add_argument(
        "--n_mels",
        type=int,
        default=80,
        help="メルフィルタバンクの数。出力する周波数チャンネル数（デフォルト：80）",
    )
    parser.add_argument(
        "--n_fft",
        type=int,
        default=2048,
        help="FFTのポイント数。周波数解像度を決定（デフォルト：2048）",
    )
    parser.add_argument(
        "--win_length",
        type=int,
        default=1024,
        help="ウィンドウ長（サンプル数）。短時間フーリエ変換の窓長（デフォルト：1024）",
    )
    parser.add_argument(
        "--hop_length",
        type=int,
        default=256,
        help="ホップ長（サンプル数）。フレーム間の重複を制御（デフォルト：256）",
    )
    parser.add_argument(
        "--fmin",
        type=float,
        default=125,
        help="最低周波数（Hz）。解析する周波数の下限（デフォルト：125）",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=12000,
        help="最高周波数（Hz）。解析する周波数の上限（デフォルト：12000）",
    )
    parser.add_argument(
        "--min_level",
        type=float,
        default=1e-5,
        help="最小レベル値。対数変換時の下限値（デフォルト：1e-5）",
    )
    parser.add_argument(
        "--max_level",
        type=float,
        help="最大レベル値。正規化時の上限値（デフォルト：自動計算）",
    )
    parser.add_argument(
        "--disable_normalize",
        action="store_true",
        help="正規化を無効にする。指定すると正規化処理をスキップします",
    )
    extract_melspectrogram(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
