from pathlib import Path

from syrupy.assertion import SnapshotAssertion

from acoustic_feature_extractor.data.f0 import F0
from extractor.extract_converted_f0 import extract_converted_f0


def test_extract_converted_f0(
    data_dir: Path,
    with_vuv: bool,
    snapshot_json: SnapshotAssertion,
):
    output_dir = data_dir / f"output_extract_converted_f0-with_vuv={with_vuv}"

    input_stat = (
        data_dir
        / "output_extract_f0_statistics_lowhigh"
        / f"statistics_low-with_vuv={with_vuv}.npy"
    )
    target_stat = (
        data_dir
        / "output_extract_f0_statistics_lowhigh"
        / f"statistics_high-with_vuv={with_vuv}.npy"
    )

    extract_converted_f0(
        input_glob=data_dir
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

    output_data = list(map(F0.load, sorted(output_dir.glob("*.npy"))))
    result = []
    for f0_data in output_data:
        result.append({"array": f0_data.array.tolist(), "rate": f0_data.rate})

    assert result == snapshot_json
