import numpy
import pytest

from acoustic_feature_extractor.data.f0 import F0


@pytest.fixture(params=(True, False))
def with_vuv(request):
    return request.param


def test_f0_convert(
        with_vuv: bool,
):
    frequency = numpy.array([0, 100, 200, 300], dtype=numpy.float32)

    f0 = F0.from_frequency(
        frequency=frequency,
        frame_period=5,
        with_vuv=with_vuv,
    ).convert(
        input_mean=numpy.log(1),
        input_var=1,
        target_mean=numpy.log(2),
        target_var=1,
    )

    if with_vuv:
        expected_frequency = numpy.array([200, 200, 400, 600], dtype=numpy.float32)
    else:
        expected_frequency = numpy.array([0, 200, 400, 600], dtype=numpy.float32)

    converted_frequency = f0.array.reshape(len(f0.array), -1)[:, 0]
    converted_frequency[converted_frequency != 0] = numpy.exp(converted_frequency[converted_frequency != 0])
    numpy.testing.assert_allclose(converted_frequency, expected_frequency, rtol=0, atol=1e-4)
