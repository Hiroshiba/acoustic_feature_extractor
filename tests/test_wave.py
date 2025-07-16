from pathlib import Path

import librosa

from acoustic_feature_extractor.data.wave import Wave


def test_wave_load():
    Wave.load(Path(librosa.example("brahms")))


def test_wave_save():
    Wave.load(Path(librosa.example("brahms"))).save(Path("/tmp/sample.wav"))
