import numpy
import pytest
from syrupy.assertion import SnapshotAssertion

from acoustic_feature_extractor.data.f0 import F0


@pytest.fixture(params=(True, False))
def with_vuv(request):
    return request.param


def test_f0_convert(with_vuv: bool, snapshot_json: SnapshotAssertion):
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

    converted_frequency = f0.array.reshape(len(f0.array), -1)[:, 0]
    converted_frequency[converted_frequency != 0] = numpy.exp(
        converted_frequency[converted_frequency != 0]
    )

    result = {"converted_frequency": converted_frequency.tolist(), "with_vuv": with_vuv}

    assert result == snapshot_json
