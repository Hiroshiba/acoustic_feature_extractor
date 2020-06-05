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

        from extractor import extract_phoneme

        dir(extract_phoneme)

        from extractor import extract_silence

        dir(extract_silence)

        from extractor import extract_volume

        dir(extract_volume)

        from extractor import extract_wave

        dir(extract_wave)

    def test_import_data(self):
        from acoustic_feature_extractor.data import linguistic_feature

        dir(linguistic_feature)

        from acoustic_feature_extractor.data import phoneme

        dir(phoneme)

        from acoustic_feature_extractor.data import sampling_data

        dir(sampling_data)

        from acoustic_feature_extractor.data import spectrogram

        dir(spectrogram)

        from acoustic_feature_extractor.data import wave

        dir(wave)

    def test_import_utility(self):
        from acoustic_feature_extractor.utility import json_utility

        dir(json_utility)
