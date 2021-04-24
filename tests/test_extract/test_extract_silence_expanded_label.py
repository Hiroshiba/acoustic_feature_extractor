from pathlib import Path

from acoustic_feature_extractor.data.phoneme import JvsPhoneme, PhonemeType
from extractor.extract_silence_expanded_label import extract_silence_expanded_label
from tests.utility import data_base_dir, true_data_base_dir


def test_extract_silence_expanded_label(data_dir: Path):
    output_dir = data_dir / "output_extract_silence_expanded_label"
    extract_silence_expanded_label(
        input_label_glob=str(data_base_dir / "phoneme/*.txt"),
        input_silence_glob=str(data_base_dir / "silence/*.npy"),
        output_directory=output_dir,
        phoneme_type=PhonemeType.jvs,
    )

    true_data_dir = true_data_base_dir.joinpath("output_extract_silence_expanded_label")

    output_paths = sorted(output_dir.glob("*.lab"))
    true_paths = sorted(true_data_dir.glob("*.lab"))

    # # overwrite true data
    # true_data_dir.mkdir(exist_ok=True)
    # for output_path in output_paths:
    #     output_data = JvsPhoneme.load_julius_list(output_path)
    #     true_data_dir.mkdir(parents=True, exist_ok=True)
    #     true_path = true_data_dir.joinpath(output_path.name)
    #     JvsPhoneme.save_julius_list(output_data, true_path)

    assert len(output_paths) == len(true_paths)
    for output_path, true_path in zip(output_paths, true_paths):
        output_data = JvsPhoneme.load_julius_list(output_path)
        true_data = JvsPhoneme.load_julius_list(true_path)

        assert output_data == true_data
