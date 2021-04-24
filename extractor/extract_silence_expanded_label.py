import argparse
import glob
import multiprocessing
from functools import partial
from pathlib import Path
from typing import Tuple

import numpy
import tqdm
from acoustic_feature_extractor.data.phoneme import (
    JvsPhoneme,
    PhonemeType,
    phoneme_type_to_class,
)
from acoustic_feature_extractor.data.sampling_data import SamplingData
from acoustic_feature_extractor.utility.json_utility import save_arguments


def process(
    paths: Tuple[Path, Path],
    output_directory: Path,
    phoneme_type: PhonemeType,
):
    label_path, silence_path = paths

    phoneme_class = phoneme_type_to_class[phoneme_type]
    label = phoneme_class.load_julius_list(label_path)

    silence = SamplingData.load(silence_path)
    silence_array = numpy.squeeze(silence.array)

    for silence_start, silence_end in zip(
        numpy.where(
            numpy.logical_and(numpy.diff(numpy.r_[False, silence_array]), silence_array)
        )[0]
        / silence.rate,
        numpy.where(
            numpy.logical_and(numpy.diff(numpy.r_[silence_array, False]), silence_array)
        )[0]
        / silence.rate,
    ):
        for i, l in enumerate(label):
            if l.phoneme != JvsPhoneme.space_phoneme:
                continue

            if silence_start < l.start and l.start <= silence_end:
                l.start = silence_start
                if i > 0:
                    label[i - 1].end = silence_start

            if silence_start <= l.end and l.end < silence_end:
                l.end = silence_end
                if i < len(label) - 1:
                    label[i + 1].start = silence_end

    out = output_directory / (label_path.stem + ".lab")
    phoneme_class.save_julius_list(phonemes=label, path=out)


def extract_silence_expanded_label(
    input_label_glob: str,
    input_silence_glob: str,
    output_directory: Path,
    phoneme_type: PhonemeType,
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
    )

    with multiprocessing.Pool() as pool:
        list(
            tqdm.tqdm(
                pool.imap_unordered(_process, zip(label_paths, silence_paths)),
                total=len(label_paths),
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_label_glob", "-ilg", required=True)
    parser.add_argument("--input_silence_glob", "-isg", required=True)
    parser.add_argument("--output_directory", "-od", type=Path, required=True)
    parser.add_argument(
        "--phoneme_type", "-pt", type=PhonemeType, default=PhonemeType.seg_kit
    )
    extract_silence_expanded_label(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
