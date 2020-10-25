from pathlib import Path

import numpy
from acoustic_feature_extractor.data.f0 import F0, F0Type
from extractor.extract_f0 import extract_f0
from tests.utility import true_data_base_dir


def test_extract_f0(
    data_dir: Path,
    with_vuv: bool,
    f0_type: F0Type,
):
    output_dir = data_dir / f"output_extract_f0-with_vuv={with_vuv}-f0_type={f0_type}"
    extract_f0(
        input_glob=data_dir / "music*.wav",
        output_directory=output_dir,
        sampling_rate=24000,
        frame_period=5.0,
        f0_floor=71.0,
        f0_ceil=800.0,
        with_vuv=with_vuv,
        f0_type=f0_type,
    )

    true_data_dir = true_data_base_dir.joinpath(
        f"output_extract_f0-with_vuv={with_vuv}-f0_type={f0_type}"
    )

    output_paths = sorted(output_dir.glob("*.npy"))
    true_paths = sorted(true_data_dir.glob("*.npy"))

    # # overwrite true data
    # for output_path in output_paths:
    #     output_data = F0.load(output_path)

    #     true_data_dir.mkdir(parents=True, exist_ok=True)
    #     true_path = true_data_dir.joinpath(output_path.name)
    #     output_data.save(true_path)

    assert len(output_paths) == len(true_paths)
    for output_path, true_path in zip(output_paths, true_paths):
        output_data = F0.load(output_path)
        true_data = F0.load(true_path)

        numpy.testing.assert_allclose(
            output_data.array, true_data.array, rtol=0, atol=1e-6
        )
        assert output_data.rate == true_data.rate
