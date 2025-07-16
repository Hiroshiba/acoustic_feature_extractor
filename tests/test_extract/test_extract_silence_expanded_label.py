from pathlib import Path

from syrupy.assertion import SnapshotAssertion

from acoustic_feature_extractor.data.phoneme import JvsPhoneme, PhonemeType
from extractor.extract_silence_expanded_label import extract_silence_expanded_label
from tests.utility import data_base_dir


def test_extract_silence_expanded_label(
    data_dir: Path,
    snapshot_json: SnapshotAssertion,
):
    output_dir = data_dir / "output_extract_silence_expanded_label"
    extract_silence_expanded_label(
        input_label_glob=str(data_base_dir / "phoneme/*.txt"),
        input_silence_glob=str(data_base_dir / "silence/*.npy"),
        output_directory=output_dir,
        phoneme_type=PhonemeType.jvs,
        phoneme_minimum_second=0.03,
    )

    output_paths = sorted(output_dir.glob("*.lab"))
    result = []
    for output_path in output_paths:
        output_data = JvsPhoneme.load_julius_list(output_path)
        result.append(
            {
                "filename": output_path.name,
                "phonemes": [
                    {
                        "start": phoneme.start,
                        "end": phoneme.end,
                        "phoneme": phoneme.phoneme,
                    }
                    for phoneme in output_data
                ],
            }
        )

    assert result == snapshot_json
