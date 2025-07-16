from pathlib import Path

from syrupy.assertion import SnapshotAssertion

from acoustic_feature_extractor.data.f0 import F0, F0Type
from extractor.extract_f0 import extract_f0
from tests.utility import round_floats


def test_extract_f0(
    data_dir: Path,
    with_vuv: bool,
    f0_type: F0Type,
    snapshot_json: SnapshotAssertion,
):
    output_dir = (
        data_dir / f"output_extract_f0-with_vuv={with_vuv}-f0_type={f0_type.value}"
    )
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

    output_data = list(map(F0.load, sorted(output_dir.glob("*.npy"))))
    result = []
    for f0_data in output_data:
        result.append({"array": f0_data.array.tolist(), "rate": f0_data.rate})

    result = round_floats(result, 2)
    assert result == snapshot_json
