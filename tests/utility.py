from pathlib import Path

import librosa
import numpy

from acoustic_feature_extractor.data.wave import Wave

true_data_base_dir = Path(__file__).parent / 'true_data'


def generate_music_file(data_dir: Path, prefix: str, time_scale: float):
    path = data_dir / f'music_{prefix}_timescale={time_scale}.wav'
    if path.exists():
        return

    w, sampling_rate = librosa.load(librosa.util.example_audio_file(), sr=24000)
    w = librosa.resample(w, orig_sr=sampling_rate, target_sr=int(sampling_rate * time_scale))
    w = w[:sampling_rate * 5]
    wave = Wave(w, sampling_rate=sampling_rate)
    wave.save(path)


def generate_f0_file(data_dir: Path, prefix: str, frequencies: numpy.ndarray, with_vuv: bool):
    f0 = frequencies.astype(numpy.float32)
    vuv = f0 != 0

    f0_log = numpy.zeros_like(f0)
    f0_log[vuv] = numpy.log(f0[vuv])

    if not with_vuv:
        array = f0_log
    else:
        array = numpy.stack([f0_log, vuv.astype(f0_log.dtype)], axis=1)

    numpy.save(str(data_dir / f'f0_{prefix}_with_vuv={with_vuv}.npy'), dict(array=array, rate=200))
