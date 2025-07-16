import shutil
from pathlib import Path

import numpy
import pytest

from acoustic_feature_extractor.data.f0 import F0Type
from extractor.extract_f0 import extract_f0
from extractor.extract_f0_statistics import extract_f0_statistics
from tests.utility import generate_f0_file, generate_music_file


@pytest.fixture(params=(True, False))
def with_vuv(request):
    return request.param


@pytest.fixture(params=list(F0Type))
def f0_type(request):
    return request.param


@pytest.fixture(scope="session", autouse=True)
def generate_data(data_dir: Path):
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir()

    # generate music files
    generate_music_file(data_dir=data_dir, prefix="high", time_scale=1)
    generate_music_file(data_dir=data_dir, prefix="low", time_scale=2)

    # generate f0 files
    base_frequencies = numpy.linspace(80, 120, num=9)
    generate_f0_file(
        data_dir=data_dir,
        prefix="high",
        frequencies=base_frequencies * 2,
        with_vuv=True,
    )
    generate_f0_file(
        data_dir=data_dir, prefix="low", frequencies=base_frequencies / 2, with_vuv=True
    )
    generate_f0_file(
        data_dir=data_dir,
        prefix="high",
        frequencies=base_frequencies * 2,
        with_vuv=False,
    )
    generate_f0_file(
        data_dir=data_dir,
        prefix="low",
        frequencies=base_frequencies / 2,
        with_vuv=False,
    )

    # generate extracted f0 data for world type (needed for converted f0 test)
    for with_vuv in [True, False]:
        output_dir = data_dir / f"output_extract_f0-with_vuv={with_vuv}-f0_type=world"
        extract_f0(
            input_glob=data_dir / "music*.wav",
            output_directory=output_dir,
            sampling_rate=24000,
            frame_period=5.0,
            f0_floor=71.0,
            f0_ceil=800.0,
            with_vuv=with_vuv,
            f0_type=F0Type.world,
        )

    # generate statistics files
    stats_dir = data_dir / "output_extract_f0_statistics_lowhigh"
    stats_dir.mkdir()

    for with_vuv in [True, False]:
        extract_f0_statistics(
            input_glob=data_dir / f"f0_low*with_vuv={with_vuv}.npy",
            output=stats_dir / f"statistics_low-with_vuv={with_vuv}.npy",
        )
        extract_f0_statistics(
            input_glob=data_dir / f"f0_high*with_vuv={with_vuv}.npy",
            output=stats_dir / f"statistics_high-with_vuv={with_vuv}.npy",
        )
