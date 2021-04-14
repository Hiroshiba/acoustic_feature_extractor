from pathlib import Path

import numpy
from acoustic_feature_extractor.data.sampling_data import SamplingData
from extractor.extract_loudness import FieldType, extract_loudness
from tests.utility import true_data_base_dir


def test_extract_loudness(
    data_dir: Path,
):
    output_dir = data_dir / "output_extract_loudness"
    extract_loudness(
        input_glob=str(data_dir / "music*.wav"),
        output_directory=output_dir,
        sampling_rate=100,
        calibration_value=0.1,
        field_type=FieldType.free,
    )

    true_data_dir = true_data_base_dir.joinpath("output_extract_loudness")

    output_paths = sorted(output_dir.glob("*.npy"))
    true_paths = sorted(true_data_dir.glob("*.npy"))

    # # overwrite true data
    # for output_path in output_paths:
    #     output_data = SamplingData.load(output_path)
    #     true_data_dir.mkdir(parents=True, exist_ok=True)
    #     true_path = true_data_dir.joinpath(output_path.name)
    #     output_data.save(true_path)

    assert len(output_paths) == len(true_paths)
    for output_path, true_path in zip(output_paths, true_paths):
        output_data = SamplingData.load(output_path)
        true_data = SamplingData.load(true_path)

        numpy.testing.assert_allclose(
            output_data.array, true_data.array, rtol=0, atol=1e-6
        )
        assert output_data.rate == true_data.rate
