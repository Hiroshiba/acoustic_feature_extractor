import argparse
import glob
import multiprocessing
from functools import partial
from pathlib import Path

import librosa
import numpy
import tqdm

from acoustic_feature_extractor.data.wave import Wave
from acoustic_feature_extractor.utility.json_utility import save_arguments


def process(p: Path, sampling_rate: int):
    return Wave.load(p, sampling_rate).wave


def extract_silence(
        input_glob,
        output_directory: Path,
        sampling_rate: int,
        silence_top_db: float,
):
    output_directory.mkdir(exist_ok=True)
    save_arguments(locals(), output_directory / 'arguments.json')

    paths = [Path(p) for p in glob.glob(str(input_glob))]

    _process = partial(process, sampling_rate=sampling_rate)

    pool = multiprocessing.Pool()
    waves = list(tqdm.tqdm(pool.imap(_process, paths), total=len(paths)))
    lengths = [len(w) for w in waves]

    wave = numpy.concatenate(waves)
    intervals = librosa.effects.split(wave, top_db=silence_top_db)
    silence = numpy.ones(len(wave), dtype=bool)

    for s, t in intervals:
        silence[s:t] = False

    for i, (s, l) in enumerate(zip(numpy.cumsum([0] + lengths), lengths)):
        out = output_directory / (paths[i].stem + '.npy')
        numpy.save(str(out), dict(array=silence[s:s + l], rate=sampling_rate))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_glob', '-ig', required=True)
    parser.add_argument('--output_directory', '-od', type=Path, required=True)
    parser.add_argument('--sampling_rate', '-sr', type=int, required=True)
    parser.add_argument('--silence_top_db', '-st', type=float, default=60)
    extract_silence(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
