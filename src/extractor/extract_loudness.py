import argparse
import glob
import multiprocessing
from functools import partial
from pathlib import Path

import tqdm
from mosqito.sq_metrics.loudness.loudness_zwtv.loudness_zwtv import loudness_zwtv

from acoustic_feature_extractor.data.processing_enums import FieldType
from acoustic_feature_extractor.data.sampling_data import DegenerateType, SamplingData
from acoustic_feature_extractor.data.wave import Wave
from acoustic_feature_extractor.utility.json_utility import save_arguments


def process(
    path: Path,
    output_directory: Path,
    sampling_rate: int,
    calibration_value: float,
    field_type: FieldType,
):
    rate = 500
    assert rate % sampling_rate == 0

    wave = Wave.load(path, sampling_rate=48000)
    wave_array = wave.wave * calibration_value
    loudness = loudness_zwtv(
        signal=wave_array,
        fs=wave.sampling_rate,
        field_type=field_type.value,
    )[0]

    data = SamplingData(array=loudness, rate=rate)

    data.degenerate(
        frame_length=int(data.rate // sampling_rate),
        hop_length=int(data.rate // sampling_rate),
        centering=True,
        padding_value=0,
        padding_mode="constant",
        degenerate_type=DegenerateType.median,
    ).save(output_directory / (path.stem + ".npy"))


def extract_loudness(
    input_glob: str,
    output_directory: Path,
    sampling_rate: int,
    calibration_value: float,
    field_type: FieldType,
):
    output_directory.mkdir(exist_ok=True)
    save_arguments(locals(), output_directory / "arguments.json")

    paths = [Path(p) for p in glob.glob(str(input_glob))]
    _process = partial(
        process,
        output_directory=output_directory,
        sampling_rate=sampling_rate,
        calibration_value=calibration_value,
        field_type=field_type,
    )

    with multiprocessing.Pool() as pool:
        list(tqdm.tqdm(pool.imap(_process, paths), total=len(paths)))


def main():
    parser = argparse.ArgumentParser(
        description="音声ファイルから音の大きさ（ラウドネス）を抽出します。EBU R128準拠のアルゴリズムを使用します。"
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
        help="抽出されたラウドネスデータを保存するディレクトリ",
    )
    parser.add_argument(
        "--sampling_rate",
        "-sr",
        type=int,
        required=True,
        help="出力データのサンプリングレート（Hz）。内部的には48000Hzで処理され、指定したレートに変換されます",
    )
    parser.add_argument(
        "--calibration_value",
        "-cv",
        type=float,
        default=1,
        help="音声の校正値。音声信号に掛ける係数（デフォルト：1）",
    )
    parser.add_argument(
        "--field_type",
        "-ft",
        type=FieldType,
        default=FieldType.diffuse,
        help="音場の種類。free: 自由音場、diffuse: 拡散音場（デフォルト：diffuse）",
    )
    extract_loudness(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
