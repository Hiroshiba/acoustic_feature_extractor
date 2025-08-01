from collections.abc import Sequence
from enum import Enum

import numpy

from acoustic_feature_extractor.data.phoneme import BasePhoneme


class LinguisticFeatureType(str, Enum):
    PHONEME = "PHONEME"
    PRE_PHONEME = "PRE_PHONEME"
    POST_PHONEME = "POST_PHONEME"
    PHONEME_ID = "PHONEME_ID"
    PHONEME_DURATION = "PHONEME_DURATION"
    PRE_PHONEME_DURATION = "PRE_PHONEME_DURATION"
    POST_PHONEME_DURATION = "POST_PHONEME_DURATION"
    ACCENT = "ACCENT"
    POS_IN_PHONEME = "POS_IN_PHONEME"

    def is_phoneme(self):
        return self in (
            self.PHONEME,
            self.PRE_PHONEME,
            self.POST_PHONEME,
            self.PHONEME_ID,
            self.PHONEME_DURATION,
            self.PRE_PHONEME_DURATION,
            self.POST_PHONEME_DURATION,
            self.ACCENT,
        )


class LinguisticFeature:
    def __init__(
        self,
        phonemes: list[BasePhoneme],
        phoneme_class: type[BasePhoneme],
        rate: int,
        feature_types: Sequence[LinguisticFeatureType | str],
        start_accents: Sequence[bool] | None = None,
        end_accents: Sequence[bool] | None = None,
    ):
        if start_accents is not None:
            assert len(start_accents) == len(phonemes)
        if end_accents is not None:
            assert len(end_accents) == len(phonemes)

        self.phonemes = phonemes
        self.phoneme_class = phoneme_class
        self.rate = rate
        self.types = [LinguisticFeatureType(t) for t in feature_types]
        self.start_accents = start_accents
        self.end_accents = end_accents

    def get_dim(self, t: LinguisticFeatureType) -> int:
        return {
            t.PHONEME: self.phoneme_class.num_phoneme,
            t.PRE_PHONEME: self.phoneme_class.num_phoneme,
            t.POST_PHONEME: self.phoneme_class.num_phoneme,
            t.PHONEME_ID: 1,
            t.PHONEME_DURATION: 1,
            t.PRE_PHONEME_DURATION: 1,
            t.POST_PHONEME_DURATION: 1,
            t.ACCENT: 2,
            t.POS_IN_PHONEME: 2,
        }[t]

    def sum_dims(self, types: list[LinguisticFeatureType]):
        return sum(self.get_dim(t) for t in types)

    def _to_index(self, t: float):
        return int(round(t * self.rate))

    def _to_time(self, i: int | numpy.ndarray):
        return i / self.rate

    @property
    def len_array(self):
        return self._to_index(self.phonemes[-1].end) + 1

    def _get_phoneme(self, i: int):
        if 0 <= i < len(self.phonemes):
            return self.phonemes[i]
        elif i < 0:
            return self.phoneme_class(
                phoneme=self.phoneme_class.space_phoneme,
                start=self.phonemes[0].start,
                end=self.phonemes[0].start,
            )
        else:
            return self.phoneme_class(
                phoneme=self.phoneme_class.space_phoneme,
                start=self.phonemes[-1].end,
                end=self.phonemes[-1].end,
            )

    def _make_phoneme_array(self, dtype=numpy.float32):
        types = list(filter(LinguisticFeatureType.is_phoneme, self.types))

        array = numpy.zeros((len(self.phonemes), self.sum_dims(types)), dtype=dtype)
        for i in range(len(self.phonemes)):
            features = []
            for t in types:
                if t == LinguisticFeatureType.PHONEME:
                    features.append(self._get_phoneme(i).onehot)
                elif t == LinguisticFeatureType.PRE_PHONEME:
                    features.append(self._get_phoneme(i - 1).onehot)
                elif t == LinguisticFeatureType.POST_PHONEME:
                    features.append(self._get_phoneme(i + 1).onehot)
                elif t == LinguisticFeatureType.PHONEME_ID:
                    features.append(self._get_phoneme(i).phoneme_id)
                elif t == LinguisticFeatureType.PHONEME_DURATION:
                    features.append(self._get_phoneme(i).duration)
                elif t == LinguisticFeatureType.PRE_PHONEME_DURATION:
                    features.append(self._get_phoneme(i - 1).duration)
                elif t == LinguisticFeatureType.POST_PHONEME_DURATION:
                    features.append(self._get_phoneme(i + 1).duration)
                elif t == LinguisticFeatureType.ACCENT:
                    features.append(
                        [bool(self.start_accents[i]), bool(self.end_accents[i])]
                    )
                else:
                    raise ValueError(t)
            array[i] = numpy.concatenate(
                [numpy.asarray(f).reshape(1, -1) for f in features], axis=1
            )
        return array

    def make_array(self, dtype=numpy.float32):
        phoneme_array = self._make_phoneme_array(dtype=dtype)

        array = numpy.zeros((self.len_array, self.sum_dims(self.types)), dtype=dtype)
        for i, p in enumerate(self.phonemes):
            s = self._to_index(p.start)
            e = self._to_index(p.end)

            features = [
                numpy.repeat(phoneme_array[i].reshape(1, -1), repeats=e - s + 1, axis=0)
            ]

            if LinguisticFeatureType.POS_IN_PHONEME in self.types:
                pos_start = (self._to_time(numpy.arange(s, e + 1)) - p.start).reshape(
                    -1, 1
                )
                pos_end = p.duration - pos_start
                features.append(pos_start)
                features.append(pos_end)

            array[s : e + 1] = numpy.concatenate(features, axis=1)
        return array
