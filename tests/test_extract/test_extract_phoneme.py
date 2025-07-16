from pathlib import Path

import pytest

from acoustic_feature_extractor.data.phoneme import PhonemeType
from extractor.extract_phoneme import extract_phoneme
from tests.utility import data_base_dir


@pytest.fixture(params=(True, False))
def with_phoneme_id(request):
    return request.param


@pytest.fixture(params=(True, False))
def with_pre_post(request):
    return request.param


@pytest.fixture(params=(True, False))
def with_duration(request):
    return request.param


@pytest.fixture(params=(True, False))
def with_relative_pos(request):
    return request.param


def test_extract_phoneme(
    data_dir: Path,
    with_phoneme_id: bool,
    with_pre_post: bool,
    with_duration: bool,
    with_relative_pos: bool,
):
    output_dir = data_dir / (
        "output_extract_phoneme"
        f",with_phoneme_id={with_phoneme_id}"
        f",with_pre_post={with_pre_post}"
        f",with_duration={with_duration}"
        f",with_relative_pos={with_relative_pos}"
    )
    extract_phoneme(
        input_glob=str(data_base_dir / "phoneme/*.txt"),
        output_directory=output_dir,
        input_start_accent_glob=None,
        input_end_accent_glob=None,
        phoneme_type=PhonemeType.jvs,
        with_phoneme_id=with_phoneme_id,
        with_pre_post=with_pre_post,
        with_duration=with_duration,
        with_relative_pos=with_relative_pos,
        rate=100,
        ignore_error=False,
    )


def test_extract_phoneme_with_accent(
    data_dir: Path,
    with_phoneme_id: bool,
    with_pre_post: bool,
    with_duration: bool,
    with_relative_pos: bool,
):
    output_dir = data_dir / (
        "test_extract_phoneme_with_accent"
        f",with_phoneme_id={with_phoneme_id}"
        f",with_pre_post={with_pre_post}"
        f",with_duration={with_duration}"
        f",with_relative_pos={with_relative_pos}"
    )
    extract_phoneme(
        input_glob=str(data_base_dir / "phoneme/*.txt"),
        output_directory=output_dir,
        input_start_accent_glob=str(data_base_dir / "start_accent/*.txt"),
        input_end_accent_glob=str(data_base_dir / "end_accent/*.txt"),
        phoneme_type=PhonemeType.jvs,
        with_phoneme_id=with_phoneme_id,
        with_pre_post=with_pre_post,
        with_duration=with_duration,
        with_relative_pos=with_relative_pos,
        rate=100,
        ignore_error=False,
    )
