from pathlib import Path

from syrupy.assertion import SnapshotAssertion

from acoustic_feature_extractor.utility.numpy_utility import load_numpy_object
from extractor.extract_f0_statistics import extract_f0_statistics
from tests.utility import round_floats


def test_extract_f0_statistics(
    data_dir: Path,
    with_vuv: bool,
    snapshot_json: SnapshotAssertion,
):
    output_dir = data_dir / "output_extract_f0_statistics"
    output_dir.mkdir(exist_ok=True)

    extract_f0_statistics(
        input_glob=data_dir / f"f0*with_vuv={with_vuv}.npy",
        output=output_dir / f"statistics-with_vuv={with_vuv}.npy",
    )

    statistics = load_numpy_object(output_dir / f"statistics-with_vuv={with_vuv}.npy")

    result = {
        "with_vuv": with_vuv,
        "mean": float(statistics["mean"]),
        "var": float(statistics["var"]),
    }

    result = round_floats(result, 2)
    assert result == snapshot_json


def test_extract_f0_statistics_lowhigh(
    data_dir: Path,
    with_vuv: bool,
    snapshot_json: SnapshotAssertion,
):
    output_dir = data_dir / "output_extract_f0_statistics_lowhigh"
    output_dir.mkdir(exist_ok=True)

    extract_f0_statistics(
        input_glob=data_dir / f"f0_low*with_vuv={with_vuv}.npy",
        output=output_dir / f"statistics_low-with_vuv={with_vuv}.npy",
    )
    statistics_low = load_numpy_object(
        output_dir / f"statistics_low-with_vuv={with_vuv}.npy"
    )

    extract_f0_statistics(
        input_glob=data_dir / f"f0_high*with_vuv={with_vuv}.npy",
        output=output_dir / f"statistics_high-with_vuv={with_vuv}.npy",
    )
    statistics_high = load_numpy_object(
        output_dir / f"statistics_high-with_vuv={with_vuv}.npy"
    )

    result = {
        "with_vuv": with_vuv,
        "statistics_low": {
            "mean": float(statistics_low["mean"]),
            "var": float(statistics_low["var"]),
        },
        "statistics_high": {
            "mean": float(statistics_high["mean"]),
            "var": float(statistics_high["var"]),
        },
        "mean_comparison": statistics_low["mean"] < statistics_high["mean"],
    }

    result = round_floats(result, 2)
    assert result == snapshot_json
