import argparse
import glob
import multiprocessing
from functools import partial
from pathlib import Path
from typing import Sequence

import numpy
import tqdm
from acoustic_feature_extractor.data.linguistic_feature import (
    LinguisticFeature,
    LinguisticFeatureType,
)
from acoustic_feature_extractor.data.phoneme import PhonemeType, phoneme_type_to_class
from acoustic_feature_extractor.utility.json_utility import save_arguments


def process(
    path: Path,
    output_directory: Path,
    phoneme_type: PhonemeType,
    rate: int,
    types: Sequence[LinguisticFeatureType],
):
    phoneme_class = phoneme_type_to_class[phoneme_type]
    ps = phoneme_class.load_julius_list(path)
    array = LinguisticFeature(
        phonemes=ps, phoneme_class=phoneme_class, rate=rate, feature_types=types
    ).make_array()

    out = output_directory / (path.stem + ".npy")
    numpy.save(str(out), dict(array=array, rate=rate))


def extract_phoneme(
    input_glob,
    output_directory: Path,
    phoneme_type: PhonemeType,
    with_pre_post: bool,
    with_duration: bool,
    with_relative_pos: bool,
    rate: int,
):
    output_directory.mkdir(exist_ok=True)
    save_arguments(locals(), output_directory / "arguments.json")

    # Linguistic Feature Type
    types = [LinguisticFeatureType.PHONEME]

    if with_pre_post:
        types += [
            LinguisticFeatureType.PRE_PHONEME,
            LinguisticFeatureType.POST_PHONEME,
        ]

    if with_duration:
        types += [LinguisticFeatureType.PHONEME_DURATION]

        if with_pre_post:
            types += [
                LinguisticFeatureType.PRE_PHONEME_DURATION,
                LinguisticFeatureType.POST_PHONEME_DURATION,
            ]

    if with_relative_pos:
        types += [LinguisticFeatureType.POS_IN_PHONEME]

    print("types:", [t.value for t in types])

    paths = [Path(p) for p in glob.glob(str(input_glob))]
    _process = partial(
        process,
        output_directory=output_directory,
        phoneme_type=phoneme_type,
        rate=rate,
        types=types,
    )

    pool = multiprocessing.Pool()
    list(tqdm.tqdm(pool.imap_unordered(_process, paths), total=len(paths)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glob", "-ig", required=True)
    parser.add_argument("--output_directory", "-od", type=Path, required=True)
    parser.add_argument(
        "--phoneme_type", "-pt", type=PhonemeType, default=PhonemeType.seg_kit
    )
    parser.add_argument("--with_pre_post", "-wpp", action="store_true")
    parser.add_argument("--with_duration", "-wd", action="store_true")
    parser.add_argument("--with_relative_pos", "-wrp", action="store_true")
    parser.add_argument("--rate", "-r", type=int, default=100)
    extract_phoneme(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
