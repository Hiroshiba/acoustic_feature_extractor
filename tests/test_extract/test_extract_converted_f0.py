from pathlib import Path

import numpy
from acoustic_feature_extractor.data.f0 import F0
from extractor.extract_converted_f0 import extract_converted_f0
from tests.utility import true_data_base_dir


def test_extract_converted_f0(
    data_dir: Path,
    with_vuv: bool,
):
    output_dir = data_dir / f"output_extract_converted_f0-with_vuv={with_vuv}"

    input_stat = (
        true_data_base_dir
        / "output_extract_f0_statistics_lowhigh"
        / f"statistics_low-with_vuv={with_vuv}.npy"
    )
    target_stat = (
        true_data_base_dir
        / "output_extract_f0_statistics_lowhigh"
        / f"statistics_high-with_vuv={with_vuv}.npy"
    )

    extract_converted_f0(
        input_glob=true_data_base_dir
        / f"output_extract_f0-with_vuv={with_vuv}-f0_type=world"
        / "*.npy",
        output_directory=output_dir,
        input_statistics=input_stat,
        target_statistics=target_stat,
        input_mean=None,
        input_var=None,
        target_mean=None,
        target_var=None,
    )

    true_data_dir = (
        true_data_base_dir / f"output_extract_converted_f0-with_vuv={with_vuv}"
    )

    output_data = list(map(F0.load, sorted(output_dir.glob("*.npy"))))
    true_data = list(map(F0.load, sorted(true_data_dir.glob("*.npy"))))
    assert len(output_data) == len(true_data)

    for output_datum, true_datum in zip(output_data, true_data):
        numpy.testing.assert_allclose(
            output_datum.array, true_datum.array, rtol=0, atol=1e-6
        )
        assert output_datum.rate == true_datum.rate
