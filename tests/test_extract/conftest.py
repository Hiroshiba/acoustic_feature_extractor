import shutil
from pathlib import Path

import numpy
import pytest

from tests.utility import generate_music_file, generate_f0_file


@pytest.fixture(params=(True, False))
def with_vuv(request):
    return request.param


@pytest.fixture(scope="session", autouse=True)
def generate_data(data_dir: Path):
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir()

    # generate music files
    generate_music_file(data_dir=data_dir, prefix='high', time_scale=1)
    generate_music_file(data_dir=data_dir, prefix='low', time_scale=2)

    # generate f0 files
    base_frequencies = numpy.linspace(80, 120, num=9)
    generate_f0_file(data_dir=data_dir, prefix='high', frequencies=base_frequencies * 2, with_vuv=True)
    generate_f0_file(data_dir=data_dir, prefix='low', frequencies=base_frequencies / 2, with_vuv=True)
    generate_f0_file(data_dir=data_dir, prefix='high', frequencies=base_frequencies * 2, with_vuv=False)
    generate_f0_file(data_dir=data_dir, prefix='low', frequencies=base_frequencies / 2, with_vuv=False)
