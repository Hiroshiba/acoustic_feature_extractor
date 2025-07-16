from pathlib import Path

from syrupy.assertion import SnapshotAssertion

from acoustic_feature_extractor.data.sampling_data import SamplingData
from extractor.extract_loudness import FieldType, extract_loudness
from tests.utility import round_floats


def test_extract_loudness(
    data_dir: Path,
    snapshot_json: SnapshotAssertion,
):
    output_dir = data_dir / "output_extract_loudness"
    extract_loudness(
        input_glob=str(data_dir / "music*.wav"),
        output_directory=output_dir,
        sampling_rate=100,
        calibration_value=0.1,
        field_type=FieldType.free,
    )

    output_data = list(map(SamplingData.load, sorted(output_dir.glob("*.npy"))))
    result = []
    for sampling_data in output_data:
        result.append(
            {"array": sampling_data.array.tolist(), "rate": sampling_data.rate}
        )

    result = round_floats(result, 2)
    assert result == snapshot_json
