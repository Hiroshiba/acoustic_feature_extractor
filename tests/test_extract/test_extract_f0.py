from pathlib import Path

import numpy
import pytest

from acoustic_feature_extractor.data.f0 import F0
from extractor.extract_f0 import extract_f0
from tests.utility import true_data_base_dir


def test_extract_f0(
        data_dir: Path,
        with_vuv: bool,
):
    output_dir = data_dir / f'output_extract_f0-with_vuv={with_vuv}'
    extract_f0(
        input_glob=data_dir / 'music*.wav',
        output_directory=output_dir,
        sampling_rate=24000,
        frame_period=5.0,
        f0_floor=71.0,
        f0_ceil=800.0,
        with_vuv=with_vuv,
    )

    true_data_dir = true_data_base_dir / f'output_extract_f0-with_vuv={with_vuv}'

    output_data = list(map(F0.load, sorted(output_dir.glob('*.npy'))))
    true_data = list(map(F0.load, sorted(true_data_dir.glob('*.npy'))))
    assert len(output_data) == len(true_data)

    for output_datum, true_datum in zip(output_data, true_data):
        numpy.testing.assert_allclose(output_datum.array, true_datum.array, rtol=0, atol=1e-6)
        assert output_datum.rate == true_datum.rate
