import argparse
import glob
import multiprocessing
from functools import partial
from pathlib import Path

import numpy
import tqdm

from acoustic_feature_extractor.data.spectrogram import to_log_melspectrogram
from acoustic_feature_extractor.data.wave import Wave


def process(
    path: Path,
    sampling_rate: int,
    preemph: float | None,
    n_mels: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    fmin: float,
    fmax: float,
    eliminate_silence: bool,
):
    wave = Wave.load(path, sampling_rate)

    min_level = 1e-15
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
        max_level=None,
        normalize=False,
    )

    if eliminate_silence:
        ms[ms == min_level] = ms[ms != min_level].min()

    return ms.min(), ms.max()


def analyze_melspectrogram(
    input_glob,
    sampling_rate: int,
    preemph: float | None,
    n_mels: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    fmin: float,
    fmax: float,
    eliminate_silence: bool,
):
    paths = [Path(p) for p in glob.glob(str(input_glob))]
    _process = partial(
        process,
        sampling_rate=sampling_rate,
        preemph=preemph,
        n_mels=n_mels,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        eliminate_silence=eliminate_silence,
    )

    pool = multiprocessing.Pool()
    datas = list(
        tqdm.tqdm(
            pool.imap(_process, paths), total=len(paths), desc="analyze_melspectrogram"
        )
    )

    m = numpy.array(datas)
    print("min", m[:, 0].min())
    print("max", m[:, 1].max())


def main():
    parser = argparse.ArgumentParser(
        description="音声ファイルからメルスペクトログラムを分析し、最大値・最小値を表示します。正規化パラメータの決定に使用します。"
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
        "--eliminate_silence",
        action="store_true",
        help="無音部分を除去するかどうか。指定すると無音部分を除いて解析します",
    )
    analyze_melspectrogram(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
