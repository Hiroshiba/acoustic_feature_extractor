from pathlib import Path

import numpy

from acoustic_feature_extractor.utility.numpy_utility import load_numpy_object
from extractor.extract_f0_statistics import extract_f0_statistics


def test_extract_f0_statistics(
        data_dir: Path,
        with_vuv: bool,
):
    output_dir = data_dir / 'output_extract_f0_statistics'
    output_dir.mkdir(exist_ok=True)

    extract_f0_statistics(
        input_glob=data_dir / f'f0*with_vuv={with_vuv}.npy',
        output=output_dir / f'statistics-with_vuv={with_vuv}.npy',
        with_vuv=with_vuv,
    )


def test_extract_f0_statistics_lowhigh(
        data_dir: Path,
        with_vuv: bool,
):
    output_dir = data_dir / 'output_extract_f0_statistics_lowhigh'
    output_dir.mkdir(exist_ok=True)

    extract_f0_statistics(
        input_glob=data_dir / f'f0_low*with_vuv={with_vuv}.npy',
        output=output_dir / f'statistics_low-with_vuv={with_vuv}.npy',
        with_vuv=with_vuv,
    )
    statistics_low = load_numpy_object(output_dir / f'statistics_low-with_vuv={with_vuv}.npy')

    extract_f0_statistics(
        input_glob=data_dir / f'f0_high*with_vuv={with_vuv}.npy',
        output=output_dir / f'statistics_high-with_vuv={with_vuv}.npy',
        with_vuv=with_vuv,
    )
    statistics_high = load_numpy_object(output_dir / f'statistics_high-with_vuv={with_vuv}.npy')

    assert statistics_low['mean'] < statistics_high['mean']
    numpy.testing.assert_allclose(statistics_low['var'], statistics_high['var'], rtol=0, atol=1e-6)
