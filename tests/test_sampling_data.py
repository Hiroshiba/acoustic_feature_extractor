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
            b = numpy.repeat(sample, scale, axis=0)[index : index + length]
            numpy.testing.assert_equal(a, b)

    def test_split(self):
        sample100 = numpy.arange(100)
        sample100_rate100 = SamplingData(array=sample100, rate=100)

        seconds = numpy.array([20, 40, 60, 80]) / 100
        expected_arrays = numpy.split(sample100, 5)

        outputs = sample100_rate100.split(keypoint_seconds=seconds)
        for output, expected in zip(outputs, expected_arrays):
            self.assertTrue(numpy.all(output.array == expected))

    def test_estimate_padding_value(self):
        sample100 = numpy.arange(100)
        sample100[:5] = sample100[-5:] = 0
        self.assertEqual(
            SamplingData(array=sample100, rate=100).estimate_padding_value(), 0
        )

    def test_estimate_padding_value_assert(self):
        sample100 = numpy.arange(100)
        with self.assertRaises(BaseException):
            self.assertEqual(
                SamplingData(array=sample100, rate=100).estimate_padding_value(), 0
            )

    def test_padding(self):
        sample100 = numpy.arange(100)
        sample100[:5] = sample100[-5:] = 0

        arrays = numpy.split(sample100, [10, 30, 60])
        datas = [SamplingData(array=array, rate=100) for array in arrays]
        datas = SamplingData.padding(datas, padding_value=numpy.array(0))

        for data, array in zip(datas, arrays):
            expected_array = numpy.pad(array, pad_width=[0, 40 - len(array)])
            self.assertTrue(numpy.all(data.array == expected_array))

    def test_all_same(self):
        self.assertTrue(SamplingData(array=numpy.zeros(10), rate=100).all_same())
        self.assertFalse(SamplingData(array=numpy.arange(10), rate=100).all_same())
