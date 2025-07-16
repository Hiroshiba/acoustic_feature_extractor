import unittest
from itertools import product

import numpy
from parameterized import parameterized

from acoustic_feature_extractor.data.sampling_data import (
    ResampleInterpolateKind,
    SamplingData,
)


class TestSamplingData(unittest.TestCase):
    def test_resample_twice(self):
        sample100 = numpy.random.rand(100, 1)
        sample100_rate100 = SamplingData(array=sample100, rate=100)

        a = sample100_rate100.resample(sampling_rate=200, index=0, length=200)
        b = numpy.repeat(sample100, 2, axis=0)
        numpy.testing.assert_equal(a, b)

    def test_resample_half(self):
        sample100 = numpy.random.rand(100, 1)
        sample100_rate100 = SamplingData(array=sample100, rate=100)

        a = sample100_rate100.resample(sampling_rate=200, index=50, length=100)
        b = numpy.repeat(sample100, 2, axis=0)[50:150]
        numpy.testing.assert_equal(a, b)

    def test_resample_random(self):
        for _ in range(10):
            num = numpy.random.randint(256**2) + 1
            size = numpy.random.randint(5) + 1
            rate = numpy.random.randint(100) + 1
            scale = numpy.random.randint(100) + 1
            index = numpy.random.randint(num)
            length = numpy.random.randint(num - index)

            sample = numpy.random.rand(num, size)
            data = SamplingData(array=sample, rate=rate)

            a = data.resample(sampling_rate=rate * scale, index=index, length=length)
            b = numpy.repeat(sample, scale, axis=0)[index : index + length]
            numpy.testing.assert_equal(a, b)

    @parameterized.expand(
        product(
            [100, 200, 24000 / 512],
            [100, 200, 24000 / 512],
            list(ResampleInterpolateKind),
        )
    )
    def test_resample_float(
        self, source_rate: float, target_rate: float, kind: ResampleInterpolateKind
    ):
        length = 1000
        for _ in range(10):
            sample = numpy.arange(length, dtype=numpy.float32)
            data = SamplingData(array=sample, rate=source_rate)
            output = data.resample(target_rate, kind=kind)
            expect = numpy.interp(
                (
                    numpy.arange(
                        length * target_rate // source_rate, dtype=numpy.float32
                    )
                    * source_rate
                    / target_rate
                ),
                sample,
                sample,
            )

            assert numpy.all(
                numpy.abs(expect - output) < numpy.ceil(source_rate / target_rate)
            )

    def test_split(self):
        sample100 = numpy.arange(100)
        sample100_rate100 = SamplingData(array=sample100, rate=100)

        seconds = numpy.array([20, 40, 60, 80]) / 100
        expected_arrays = numpy.split(sample100, 5)

        outputs = sample100_rate100.split(keypoint_seconds=seconds)
        for output, expected in zip(outputs, expected_arrays, strict=False):
            self.assertTrue(numpy.all(output.array == expected))

    def test_estimate_padding_value(self):
        sample100 = numpy.arange(100)
        sample100[:5] = sample100[-5:] = 0
        self.assertEqual(
            SamplingData(array=sample100, rate=100).estimate_padding_value(), 0
        )

    def test_estimate_padding_value_assert(self):
        sample100 = numpy.arange(100)
        with self.assertRaises(AssertionError):
            self.assertEqual(
                SamplingData(array=sample100, rate=100).estimate_padding_value(), 0
            )

    def test_padding(self):
        sample100 = numpy.arange(100)
        sample100[:5] = sample100[-5:] = 0

        arrays = numpy.split(sample100, [10, 30, 60])
        datas = [SamplingData(array=array, rate=100) for array in arrays]
        datas = SamplingData.padding(datas, padding_value=numpy.array(0))

        for data, array in zip(datas, arrays, strict=False):
            expected_array = numpy.pad(array, pad_width=[0, 40 - len(array)])
            self.assertTrue(numpy.all(data.array == expected_array))

    def test_all_same(self):
        self.assertTrue(SamplingData(array=numpy.zeros(10), rate=100).all_same())
        self.assertFalse(SamplingData(array=numpy.arange(10), rate=100).all_same())
