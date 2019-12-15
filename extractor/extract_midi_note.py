import argparse
import glob
import multiprocessing
from functools import partial
from pathlib import Path
from typing import Tuple

import numpy
import tqdm
from kiritan_singing_label_reader import MidiNoteReader

from data.midi_feature import MidiFeature
from utility.json_utility import save_arguments


def process(
        path: Path,
        output_directory: Path,
        pitch_range: Tuple[int, int],
        pitch_shift: int,
        with_position: bool,
        rate: int,
):
    notes = MidiNoteReader(path).get_notes()
    for note in notes:
        note.pitch += pitch_shift

    array = MidiFeature(notes=notes, pitch_range=pitch_range, rate=rate).make_array(with_position)

    out = output_directory / (path.stem + '.npy')
    numpy.save(str(out), dict(array=array, rate=rate))


def extract_midi_note(
        input_glob,
        output_directory: Path,
        pitch_range: Tuple[int, int],
        pitch_shift: int,
        with_position: bool,
        rate: int,
):
    output_directory.mkdir(exist_ok=True)
    save_arguments(locals(), output_directory / 'arguments.json')

    paths = [Path(p) for p in glob.glob(str(input_glob))]
    _process = partial(
        process,
        output_directory=output_directory,
        pitch_range=pitch_range,
        pitch_shift=pitch_shift,
        with_position=with_position,
        rate=rate,
    )

    pool = multiprocessing.Pool()
    list(tqdm.tqdm(pool.imap_unordered(_process, paths), total=len(paths)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_glob', '-ig', required=True)
    parser.add_argument('--output_directory', '-od', type=Path, required=True)
    parser.add_argument('--pitch_range', '-pr', nargs=2, type=int, default=(53, 76))
    parser.add_argument('--pitch_shift', '-ps', type=int, default=0)
    parser.add_argument('--without_position', '-wp', action='store_true')
    parser.add_argument('--rate', '-r', type=int, default=100)
    extract_midi_note(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
