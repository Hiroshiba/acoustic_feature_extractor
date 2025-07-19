from pathlib import Path

import numpy
from syrupy.assertion import SnapshotAssertion

from acoustic_feature_extractor.data.processing_enums import VolumeType
from extractor.extract_volume import extract_volume
from tests.utility import round_floats


def test_extract_volume(
    data_dir: Path,
    volume_type: VolumeType,
    snapshot_json: SnapshotAssertion,
):
    output_dir = data_dir / f"output_extract_volume-volume_type={volume_type.value}"
    extract_volume(
        input_glob=data_dir / "music*.wav",
        output_directory=output_dir,
        sampling_rate=24000,
        frame_length=800,
        hop_length=200,
        top_db=80,
        normalize=False,
        volume_type=volume_type,
    )

    output_files = sorted(output_dir.glob("*.npy"))
    result = []
    for file_path in output_files:
        data = numpy.load(str(file_path), allow_pickle=True).item()
        result.append({"array": data["array"].tolist(), "rate": data["rate"]})

    result = round_floats(result, 3)
    assert result == snapshot_json
