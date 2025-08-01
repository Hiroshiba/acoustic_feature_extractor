from collections.abc import Sequence

import pytest

from acoustic_feature_extractor.data.linguistic_feature import (
    LinguisticFeature,
    LinguisticFeatureType,
)
from acoustic_feature_extractor.data.phoneme import JvsPhoneme
from tests.utility import data_base_dir


@pytest.fixture(
    params=(
        data_base_dir.joinpath("phonemes")
        .joinpath("voiceactoress100_phoneme_openjtalk.txt")
        .read_text()
        .strip()
        .splitlines()
    )
)
def phonemes(request):
    return JvsPhoneme.convert(
        [
            JvsPhoneme(phoneme=s, start=i / 10, end=(i + 1) / 10)
            for i, s in enumerate(request.param.split())
        ]
    )


@pytest.fixture(
    params=(
        (LinguisticFeatureType.PHONEME,),
        (
            LinguisticFeatureType.PHONEME,
            LinguisticFeatureType.PRE_PHONEME,
            LinguisticFeatureType.POST_PHONEME,
        ),
        (LinguisticFeatureType.PHONEME_ID,),
    )
)
def feature_types(request):
    return request.param


@pytest.fixture(
    params=(
        data_base_dir.joinpath("accents")
        .joinpath("voiceactoress100_accent_start.txt")
        .read_text()
        .strip()
        .splitlines()
    )
)
def start_accents(request):
    return [bool(int(s)) for s in request.param.split()]


@pytest.fixture(
    params=(
        data_base_dir.joinpath("accents")
        .joinpath("voiceactoress100_accent_end.txt")
        .read_text()
        .strip()
        .splitlines()
    )
)
def end_accents(request):
    return [bool(int(s)) for s in request.param.split()]


def test_linguistic_feature(
    phonemes: list[JvsPhoneme], feature_types: Sequence[LinguisticFeatureType]
):
    feature = LinguisticFeature(
        phonemes=phonemes,
        phoneme_class=JvsPhoneme,
        rate=100,
        feature_types=feature_types,
    )
    feature.make_array()


def test_linguistic_feature_with_accent(
    phonemes: list[JvsPhoneme],
    feature_types: Sequence[LinguisticFeatureType],
    start_accents: Sequence[bool],
    end_accents: Sequence[bool],
):
    def wrapper():
        feature = LinguisticFeature(
            phonemes=phonemes,
            phoneme_class=JvsPhoneme,
            rate=100,
            feature_types=feature_types + (LinguisticFeatureType.ACCENT,),
            start_accents=start_accents,
            end_accents=end_accents,
        )
        feature.make_array()

    if len(phonemes) == len(start_accents) and len(phonemes) == len(end_accents):
        wrapper()
    else:
        with pytest.raises(AssertionError):
            wrapper()
