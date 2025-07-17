import argparse
import glob
import multiprocessing
from functools import partial
from pathlib import Path

import librosa
import numpy
import soundfile
import tqdm

from acoustic_feature_extractor.data.wave import Wave
from acoustic_feature_extractor.utility.json_utility import save_arguments


def process(p: Path, sampling_rate: int):
    return Wave.load(p, sampling_rate)


def tool_split_wave(
    input_glob,
    output_directory: Path,
    sampling_rate: int,
    silence_top_db: float,
    min_silence_second: float,
    pad_second: float,
    prefix: str,
):
    output_directory.mkdir(exist_ok=True)
    save_arguments(locals(), output_directory / "arguments.json")

    paths = [Path(p) for p in glob.glob(str(input_glob))]

    _process = partial(process, sampling_rate=sampling_rate)

    with multiprocessing.Pool() as pool:
        waves = list(tqdm.tqdm(pool.imap_unordered(_process, paths), total=len(paths)))

    if sampling_rate is None:
        sampling_rate = waves[0].sampling_rate
        for wave in waves:
            assert wave.sampling_rate == sampling_rate

    cumsum_lengths = numpy.cumsum([len(w.wave) for w in waves])

    wave = numpy.concatenate([w.wave for w in waves])
    intervals = librosa.effects.split(wave, top_db=silence_top_db)

    pad_length = int(pad_second * sampling_rate)

    # split with silent position
    starts: list[int] = []
    ends: list[int] = []
    for i, (start, end) in enumerate(intervals.tolist()):
        if len(starts) == len(ends):
            start = max(start - pad_length, 0)
            starts.append(start)

        if (
            i == len(intervals) - 1
            or intervals[i + 1][0] - end > min_silence_second * sampling_rate
        ):
            end = min(end + pad_length, len(wave) - 1)
            ends.append(end)

    # split with base wave file
    old_starts, old_ends = starts, ends
    starts = []
    ends = []
    for start, end in zip(old_starts, old_ends, strict=False):
        starts.append(start)
        for c in cumsum_lengths:
            if start + pad_length * 2 < c and c < end - pad_length * 2:  # widely trim
                ends.append(c)
                starts.append(c)
        ends.append(end)

    for i, (s, e) in enumerate(zip(starts, ends, strict=False)):
        out = output_directory / f"{prefix}{i}.wav"
        soundfile.write(out, wave[s:e], sampling_rate)


def main():
    parser = argparse.ArgumentParser(
        description="音声ファイルを無音部分で分割します。長い音声ファイルを無音部分で自動分割し、複数の短いファイルに分けます。"
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
        help="分割された音声ファイルを保存するディレクトリ",
    )
    parser.add_argument(
        "--sampling_rate",
        "-sr",
        type=int,
        help="サンプリングレート（Hz）。指定しない場合は元ファイルのサンプリングレートを使用",
    )
    parser.add_argument(
        "--silence_top_db",
        "-st",
        type=float,
        default=60,
        help="無音判定の闾値（dB）。この値より小さい音量を無音とする（デフォルト：60dB）",
    )
    parser.add_argument(
        "--min_silence_second",
        "-mss",
        type=float,
        default=0.45,
        help="分割するための最小無音継続時間（秒）。この時間以上の無音があると分割します（デフォルト：0.45秒）",
    )
    parser.add_argument(
        "--pad_second",
        "-ps",
        type=float,
        default=0.3,
        help="分割点の前後に追加する時間（秒）。分割時の緩衝時間（デフォルト：0.3秒）",
    )
    parser.add_argument(
        "--prefix",
        "-p",
        default="",
        help="出力ファイル名の接頭辞。出力ファイル名の開始部分に付ける文字列（デフォルト：なし）",
    )
    tool_split_wave(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
