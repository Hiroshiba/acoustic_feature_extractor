import random
import tempfile
from pathlib import Path

import numpy
import pytest

from acoustic_feature_extractor.data.phoneme import OjtPhoneme


def test_save_and_load():
    phonemes = []

    start = 0
    for _ in range(100):
        end = start + numpy.random.rand() + 0.01
        phonemes.append(
            OjtPhoneme(
                phoneme=random.choice(OjtPhoneme.phoneme_list), start=start, end=end
            )
        )
        start = end

    with tempfile.NamedTemporaryFile() as f:
        path = Path(f.name)

    # 正常時
    OjtPhoneme.save_julius_list(phonemes, path)
    loaded_phonemes = OjtPhoneme.load_julius_list(path)
    assert phonemes == loaded_phonemes

    # 異常時、保存時にエラー
    with pytest.raises(AssertionError):
        OjtPhoneme.save_julius_list(phonemes[::-1], path)
