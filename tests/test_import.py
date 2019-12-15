import unittest


class TestImport(unittest.TestCase):
    def test_import_analyzer(self):
        from analyzer import analyze_wave
        dir(analyze_wave)

    def test_import_extractor(self):
        from extractor import extract_collected_local
        dir(extract_collected_local)

        from extractor import extract_f0
        dir(extract_f0)

        from extractor import extract_melspectrogram
        dir(extract_melspectrogram)

        from extractor import extract_midi_note
        dir(extract_midi_note)

        from extractor import extract_phoneme
        dir(extract_phoneme)

        from extractor import extract_silence
        dir(extract_silence)

        from extractor import extract_volume
        dir(extract_volume)

        from extractor import extract_wave
        dir(extract_wave)

    def test_import_data(self):
        from data import linguistic_feature
        dir(linguistic_feature)

        from data import midi_feature
        dir(midi_feature)

        from data import phoneme
        dir(phoneme)

        from data import sampling_data
        dir(sampling_data)

        from data import spectrogram
        dir(spectrogram)

        from data import wave
        dir(wave)

    def test_import_utility(self):
        from utility import json_utility
        dir(json_utility)
