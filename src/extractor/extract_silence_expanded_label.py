import argparse
import glob
import multiprocessing
from functools import partial
from pathlib import Path

import numpy
import tqdm

from acoustic_feature_extractor.data.phoneme import PhonemeType, phoneme_type_to_class
from acoustic_feature_extractor.data.sampling_data import SamplingData
from acoustic_feature_extractor.utility.json_utility import save_arguments


def process(
    paths: tuple[Path, Path],
    output_directory: Path,
    phoneme_type: PhonemeType,
    phoneme_minimum_second: float,
):
    try:
        label_path, silence_path = paths

        phoneme_class = phoneme_type_to_class[phoneme_type]
        label = phoneme_class.load_julius_list(label_path)

        silence = SamplingData.load(silence_path)
        silence_array = numpy.squeeze(silence.array)

        for silence_start, silence_end in zip(
            numpy.where(
                numpy.logical_and(
                    numpy.diff(numpy.r_[False, silence_array]), silence_array
                )
            )[0]
            / silence.rate,
            numpy.where(
                numpy.logical_and(
                    numpy.diff(numpy.r_[silence_array, False]), silence_array
                )
            )[0]
            / silence.rate,
            strict=False,
        ):
            for i, l in enumerate(label):  # noqa: E741
                if l.phoneme != phoneme_class.space_phoneme:
                    continue

                if silence_start < l.start and l.start <= silence_end:
                    if i > 0:
                        if label[i - 1].start + phoneme_minimum_second > silence_start:
                            silence_start = label[i - 1].start + phoneme_minimum_second
                        label[i - 1].end = silence_start
                    l.start = silence_start

                if silence_start <= l.end and l.end < silence_end:
                    if i < len(label) - 1:
                        if label[i + 1].end - phoneme_minimum_second < silence_end:
                            silence_end = label[i + 1].end - phoneme_minimum_second
                        label[i + 1].start = silence_end
                    l.end = silence_end

        out = output_directory / (label_path.stem + ".lab")
        phoneme_class.save_julius_list(phonemes=label, path=out)

    except:
        print("error:", paths)
        raise


def extract_silence_expanded_label(
    input_label_glob: str,
    input_silence_glob: str,
    output_directory: Path,
    phoneme_type: PhonemeType,
    phoneme_minimum_second: float,
):
    output_directory.mkdir(exist_ok=True)
    save_arguments(locals(), output_directory / "arguments.json")

    label_paths = sorted(Path(p) for p in glob.glob(str(input_label_glob)))
    silence_paths = sorted(Path(p) for p in glob.glob(str(input_silence_glob)))
    assert len(label_paths) == len(silence_paths)

    _process = partial(
        process,
        output_directory=output_directory,
        phoneme_type=phoneme_type,
        phoneme_minimum_second=phoneme_minimum_second,
    )

    with multiprocessing.Pool() as pool:
        list(
            tqdm.tqdm(
                pool.imap_unordered(
                    _process, zip(label_paths, silence_paths, strict=False)
                ),
                total=len(label_paths),
            )
        )


def main():
    parser = argparse.ArgumentParser(
        description="音素ラベルを無音情報を使って拡張します。無音部分での音素境界を調整し、より精密な音素ラベルを生成します。"
    )
    parser.add_argument(
        "--input_label_glob",
        "-ilg",
        required=True,
        help="入力音素ラベルファイルのパスパターン（例：'*.lab'）",
    )
    parser.add_argument(
        "--input_silence_glob",
        "-isg",
        required=True,
        help="入力無音データファイルのパスパターン（例：'*.npy'）。extract_silenceで生成されたデータを指定",
    )
    parser.add_argument(
        "--output_directory",
        "-od",
        type=Path,
        required=True,
        help="拡張された音素ラベルを保存するディレクトリ",
    )
    parser.add_argument(
        "--phoneme_type",
        "-pt",
        type=PhonemeType,
        default=PhonemeType.seg_kit,
        help="音素の種類。seg_kit, jvs, openjtalk, rohan, kiritan, song, dummy（デフォルト：seg_kit）",
    )
    parser.add_argument(
        "--phoneme_minimum_second",
        "-pms",
        type=float,
        default=0.03,
        help="音素の最小継続時間（秒）。この値より短い音素は調整されます（デフォルト：0.03秒）",
    )
    extract_silence_expanded_label(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
