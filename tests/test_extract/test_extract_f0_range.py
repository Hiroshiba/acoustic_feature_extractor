import dataclasses
from pathlib import Path

from extractor.extract_f0_range import calc_lf0_statistics


def test_calc_lf0_statistics(
    data_dir: Path,
):
    input_paths = sorted(data_dir.glob("music*.wav"))

    stats = calc_lf0_statistics(
        input_paths=input_paths,
        sampling_rate=24000,
        max_num=10,
        num_loop=3,
        target_duration=30,
        verbose_dir=None,
    )

    result = dataclasses.asdict(stats)

    expected = {
        "max": 5.10,
        "mean": 4.03,
        "median": 3.73,
        "min": 3.65,
        "q001": 3.66,
        "q003": 3.67,
        "q005": 3.67,
        "q01": 3.68,
        "q99": 5.10,
        "q995": 5.10,
        "q997": 5.10,
        "q999": 5.10,
        "std": 0.38,
    }

    tolerance = 0.1
    for key, actual_value in result.items():
        expected_value = expected[key]
        error_ratio = abs(actual_value - expected_value) / expected_value
        assert error_ratio < tolerance, (
            f"{key}: expected {expected_value}, got {actual_value}, error ratio: {error_ratio:.3f}"
        )


def test_calc_lf0_statistics_with_verbose_dir(
    data_dir: Path,
    tmp_path: Path,
):
    input_paths = sorted(data_dir.glob("music*.wav"))
    verbose_dir = tmp_path / "verbose"
    verbose_dir.mkdir()

    stats = calc_lf0_statistics(
        input_paths=input_paths,
        sampling_rate=24000,
        max_num=10,
        num_loop=3,
        target_duration=30,
        verbose_dir=verbose_dir,
    )

    assert stats is not None

    plot_files = list(verbose_dir.glob("calc_lf0_statistics-*.png"))
    assert len(plot_files) == 3, f"Expected 3 plot files, got {len(plot_files)}"

    for i in range(3):
        expected_file = verbose_dir / f"calc_lf0_statistics-{i:02d}.png"
        assert expected_file.exists(), f"Expected file {expected_file} does not exist"
