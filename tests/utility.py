from pathlib import Path
from typing import Any

import librosa
import numpy

from acoustic_feature_extractor.data.wave import Wave

data_base_dir = Path(__file__).parent / "data"


def generate_music_file(data_dir: Path, prefix: str, time_scale: float):
    path = data_dir / f"music_{prefix}_timescale={time_scale}.wav"
    if path.exists():
        return

    w, sampling_rate = librosa.load(librosa.util.example("vibeace"), sr=24000)
    w = librosa.resample(
        w, orig_sr=sampling_rate, target_sr=int(sampling_rate * time_scale)
    )
    w = w[: sampling_rate * 5]
    wave = Wave(w, sampling_rate=sampling_rate)
    wave.save(path)


def generate_f0_file(
    data_dir: Path, prefix: str, frequencies: numpy.ndarray, with_vuv: bool
):
    f0 = frequencies.astype(numpy.float32)
    vuv = f0 != 0

    f0_log = numpy.zeros_like(f0)
    f0_log[vuv] = numpy.log(f0[vuv])

    if not with_vuv:
        array = f0_log
    else:
        array = numpy.stack([f0_log, vuv.astype(f0_log.dtype)], axis=1)

    numpy.save(
        str(data_dir / f"f0_{prefix}_with_vuv={with_vuv}.npy"),
        dict(array=array, rate=200),
    )


def round_floats(value: Any, round_value: int) -> Any:
    """浮動小数点数を再帰的に丸める"""
    match value:
        case float():
            return round(value, round_value)
        case numpy.ndarray() if numpy.issubdtype(value.dtype, numpy.floating):
            return numpy.round(value, round_value)
        case list():
            return [round_floats(v, round_value) for v in value]
        case dict():
            return {k: round_floats(v, round_value) for k, v in value.items()}
        case _:
            return value
