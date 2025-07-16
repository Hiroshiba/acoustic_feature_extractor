import argparse
import glob
import multiprocessing
from functools import partial
from pathlib import Path

import numpy
import tqdm
from mosqito.functions.loudness_zwicker.comp_loudness import comp_loudness

from acoustic_feature_extractor.data.wave import Wave
from acoustic_feature_extractor.data.processing_enums import FieldType


def process(
    path: Path,
    calibration_value: float,
    field_type: FieldType,
):
    wave = Wave.load(path, sampling_rate=48000)
    wave_array = wave.wave * calibration_value

    loudness = comp_loudness(
        is_stationary=False,
        signal=wave_array,
        fs=wave.sampling_rate,
        field_type=field_type.value,
    )
    return loudness["values"].max()


def analyze_loudness(
    input_glob: str,
    calibration_value: float,
    field_type: FieldType,
):
    paths = [Path(p) for p in glob.glob(str(input_glob))]
    _process = partial(
        process,
        calibration_value=calibration_value,
        field_type=field_type,
    )

    with multiprocessing.Pool() as pool:
        all_data = list(
            tqdm.tqdm(
                pool.imap(_process, paths), total=len(paths), desc="analyze_loudness"
            )
        )

    q = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    x = numpy.percentile(all_data, q=q)

    print("q", "x")
    for q_, x_ in zip(q, x, strict=False):
        print(q_, x_)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glob", "-ig", required=True)
    parser.add_argument("--calibration_value", "-cv", type=float, default=1)
    parser.add_argument(
        "--field_type", "-ft", type=FieldType, default=FieldType.diffuse
    )
    analyze_loudness(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
