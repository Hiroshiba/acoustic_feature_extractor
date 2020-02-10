import unittest

import numpy

from acoustic_feature_extractor.data.sampling_data import SamplingData


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
        for _ in range(1000):
            num = numpy.random.randint(256 ** 2) + 1
            size = numpy.random.randint(5) + 1
            rate = numpy.random.randint(100) + 1
            scale = numpy.random.randint(100) + 1
            index = numpy.random.randint(num)
            length = numpy.random.randint(num - index)

            sample = numpy.random.rand(num, size)
            data = SamplingData(array=sample, rate=rate)

            a = data.resample(sampling_rate=rate * scale, index=index, length=length)
            b = numpy.repeat(sample, scale, axis=0)[index:index + length]
            numpy.testing.assert_equal(a, b)
