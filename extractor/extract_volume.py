import argparse
import glob
import multiprocessing
from enum import Enum
from functools import partial
from pathlib import Path

import librosa
import numpy
import tqdm

from acoustic_feature_extractor.data.wave import Wave
from acoustic_feature_extractor.utility.json_utility import save_arguments


class VolumeType(str, Enum):
    rms_power = "rms_power"
    mse_power = "mse_power"
    rms_db = "rms_db"
    mse_db = "mse_db"


def process(
    path: Path,
    output_directory: Path,
    sampling_rate: int,
    frame_length: int,
    hop_length: int,
    top_db: int,
    normalize: bool,
    volume_type: VolumeType,
):
    assert sampling_rate % hop_length == 0

    w = Wave.load(path, sampling_rate).wave

    array = librosa.feature.rms(w, frame_length=frame_length, hop_length=hop_length)
    array = array.squeeze()
    if volume_type in (VolumeType.mse_power, VolumeType.mse_db):
        array = array ** 2
    if volume_type in (VolumeType.rms_db, VolumeType.mse_db):
        array = librosa.power_to_db(array, top_db=top_db)

    if normalize:
        array = numpy.clip((array - array.max()) / top_db + 1, 0, 1)

    rate = sampling_rate // hop_length

    out = output_directory / (path.stem + ".npy")
    numpy.save(str(out), dict(array=array, rate=rate))


def extract_volume(
    input_glob,
    output_directory: Path,
    sampling_rate: int,
    frame_length: int,
    hop_length: int,
    top_db: int,
    normalize: bool,
    volume_type: VolumeType,
):
    output_directory.mkdir(exist_ok=True)
    save_arguments(locals(), output_directory / "arguments.json")

    paths = [Path(p) for p in glob.glob(str(input_glob))]
    _process = partial(
        process,
        output_directory=output_directory,
        sampling_rate=sampling_rate,
        frame_length=frame_length,
        hop_length=hop_length,
        top_db=top_db,
        normalize=normalize,
        volume_type=volume_type,
    )

    pool = multiprocessing.Pool()
    list(tqdm.tqdm(pool.imap(_process, paths), total=len(paths)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glob", "-ig", required=True)
    parser.add_argument("--output_directory", "-od", type=Path, required=True)
    parser.add_argument("--sampling_rate", "-sr", type=int, required=True)
    parser.add_argument("--frame_length", "-fl", type=int, default=800)
    parser.add_argument("--hop_length", "-hl", type=int, default=200)
    parser.add_argument("--top_db", "-td", type=int, default=80)
    parser.add_argument("--normalize", "-n", action="store_true")
    parser.add_argument(
        "--volume_type", "-vt", type=VolumeType, default=VolumeType.mse_db
    )
    extract_volume(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
