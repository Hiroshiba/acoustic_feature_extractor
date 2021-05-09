import argparse
import glob
import multiprocessing
from enum import Enum
from functools import partial
from pathlib import Path

import tqdm
from acoustic_feature_extractor.data.sampling_data import DegenerateType, SamplingData
from acoustic_feature_extractor.data.wave import Wave
from acoustic_feature_extractor.utility.json_utility import save_arguments
from mosqito.functions.loudness_zwicker.comp_loudness import comp_loudness


class FieldType(str, Enum):
    free = "free"
    diffuse = "diffuse"


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
    loudness = comp_loudness(
        is_stationary=False,
        signal=wave_array,
        fs=wave.sampling_rate,
        field_type=field_type.value,
    )

    data = SamplingData(array=loudness["values"], rate=rate)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glob", "-ig", required=True)
    parser.add_argument("--output_directory", "-od", type=Path, required=True)
    parser.add_argument("--sampling_rate", "-sr", type=int, required=True)
    parser.add_argument("--calibration_value", "-cv", type=float, default=1)
    parser.add_argument(
        "--field_type", "-ft", type=FieldType, default=FieldType.diffuse
    )
    extract_loudness(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
