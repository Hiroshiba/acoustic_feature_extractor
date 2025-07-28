import argparse
import copy
import dataclasses
import json
import random
from dataclasses import dataclass
from functools import partial
from glob import glob
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import pyworld
from tqdm import tqdm


@dataclass
class LogF0Statistics:
    mean: float
    median: float
    std: float
    min: float
    max: float
    q001: float
    q003: float
    q005: float
    q01: float
    q99: float
    q995: float
    q997: float
    q999: float


def _check_sampling_rate(path: Path, sampling_rate: float):
    if librosa.get_samplerate(str(path)) != sampling_rate:
        return ValueError(
            f"サンプリングレートが異なります: {path}, {librosa.get_samplerate(str(path))} != {sampling_rate}"
        )


def _load_wave(path: Path, sampling_rate: float):
    return librosa.load(path, sr=sampling_rate, dtype=np.float64)[0]


def _get_duration(path: Path):
    return librosa.get_duration(path=path)


def _split_wave_paths(paths: list[Path], target_duration: float):
    """
    音声ファイルをtarget_duration秒ずつ分割する。
    端数は切り捨てる。
    """
    copy_paths = copy.deepcopy(paths)
    random.shuffle(copy_paths)
    del paths

    with Pool() as p:
        durations = list(p.imap(_get_duration, copy_paths))

    paths_list = []
    paths = []
    now_duration = 0
    for path, duration in zip(copy_paths, durations, strict=True):
        paths.append(path)
        now_duration += duration

        if now_duration > target_duration:
            paths_list.append(paths)
            paths = []
            now_duration = 0

    # １つしかない場合を除いて端数は切り捨てる
    if len(paths_list) == 0:
        paths_list.append(paths)

    return paths_list


def _calc_features(
    paths: list[Path], sampling_rate: float, f0_floor: float, f0_ceil: float
):
    """与えられた音声ファイルの波形を接続してf0と音量を計算する"""
    wave = np.concatenate(
        [_load_wave(path, sampling_rate=sampling_rate) for path in paths]
    )
    f0, _ = pyworld.harvest(wave, sampling_rate, f0_floor=f0_floor, f0_ceil=f0_ceil)  # type: ignore

    amp = librosa.feature.rms(y=wave, frame_length=2048, hop_length=512)[0]
    volume = librosa.amplitude_to_db(amp)

    length = int(np.floor(len(volume) / (sampling_rate / 512) * 200))
    indexes = np.arange(length) * ((sampling_rate / 512) / 200)
    volume = volume[indexes.astype(int)]

    length = min(len(f0), len(volume))
    f0 = f0[:length]
    volume = volume[:length]

    return f0, volume


def _expand_in_logspace(floor: float, ceil: float, ratio: float):
    """最小値と最大値を対数領域でratio分だけ広げる"""
    r = np.log(ceil) - np.log(floor)
    return (
        np.exp(np.log(floor) - ratio * r),
        np.exp(np.log(ceil) + ratio * r),
    )


def _weighted_percentile(
    values: np.ndarray, weights: np.ndarray, percentile: float
) -> float:
    """重み付きの分位数を計算する"""
    # ソート
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # 累積重みを計算
    cumulative_weights = np.cumsum(sorted_weights)
    total_weight = cumulative_weights[-1]

    # 分位数に対応する位置を見つける
    target_weight = percentile / 100.0 * total_weight

    # 線形補間で値を求める
    if target_weight <= cumulative_weights[0]:
        return float(sorted_values[0])
    elif target_weight >= cumulative_weights[-1]:
        return float(sorted_values[-1])
    else:
        # 補間
        idx = np.searchsorted(cumulative_weights, target_weight)
        if idx == 0:
            return float(sorted_values[0])

        # 線形補間
        w0 = cumulative_weights[idx - 1]
        w1 = cumulative_weights[idx]
        v0 = sorted_values[idx - 1]
        v1 = sorted_values[idx]

        if w1 == w0:
            return float(v0)

        alpha = (target_weight - w0) / (w1 - w0)
        return float(v0 + alpha * (v1 - v0))


def _validate_sampling_rate(
    input_paths: list[Path], sampling_rate: float | None
) -> float:
    """サンプリングレートの確認と取得"""
    if sampling_rate is None:
        sampling_rate = librosa.get_samplerate(str(input_paths[0]))
        with Pool() as p:
            it = p.imap_unordered(
                partial(_check_sampling_rate, sampling_rate=sampling_rate),
                input_paths,
                chunksize=32,
            )
            for e in filter(
                None,
                tqdm(
                    it,
                    total=len(input_paths),
                    desc="calc_lf0_statistics: check sampling rate",
                ),
            ):
                raise e
    return sampling_rate


def _subsample_paths(input_paths: list[Path], max_num: int) -> list[Path]:
    """ファイル数が多い場合は間引く"""
    if len(input_paths) <= max_num:
        return input_paths
    else:
        input_paths = sorted(input_paths)
        indexes = np.linspace(0, len(input_paths) - 1, max_num).astype(int)
        return [input_paths[i] for i in indexes]


def _filter_f0_data(
    f0: np.ndarray, volume: np.ndarray, f0_floor: float, f0_ceil: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """F0データのフィルタリング処理"""
    # unvoice領域を除外
    voiced = f0 > 0
    f0 = f0[voiced]
    volume = volume[voiced]

    # 静かな音を除外
    silence = volume < volume.max() - 45
    f0 = f0[~silence]
    volume = volume[~silence]

    # 音量を正規化
    norm_volume = (volume - volume.min()) / (volume.max() - volume.min())

    # 投票数の少ないf0を除外
    target_floor, target_ceil = _expand_in_logspace(f0_floor, f0_ceil, 0.05)
    bins = np.logspace(np.log10(target_floor), np.log10(target_ceil), 100)
    f0_hist, _ = np.histogram(f0, bins=bins, weights=norm_volume)
    index = np.digitize(f0, bins=bins)
    index[index >= len(f0_hist)] = len(f0_hist) - 1
    noised = f0_hist[index] < f0_hist.sum() * 0.003
    f0 = f0[~noised]
    volume = volume[~noised]
    norm_volume = norm_volume[~noised]

    return f0, volume, norm_volume


def _create_visualization(
    f0: np.ndarray,
    volume: np.ndarray,
    norm_volume: np.ndarray,
    f0_floor: float,
    f0_ceil: float,
    verbose_dir: Path,
    i_loop: int,
):
    """可視化処理"""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))

    plt.subplot2grid((2, 2), (0, 0))
    _ = plt.hist(f0, bins=np.logspace(np.log10(f0_floor), np.log10(f0_ceil), 100))
    plt.xscale("log")
    plt.title("f0")
    plt.axvline(f0_floor, color="r", linestyle="--")
    plt.text(f0_floor, 1, f"{f0_floor:.1f}", color="r")
    plt.axvline(f0_ceil, color="r", linestyle="--")
    plt.text(f0_ceil, 1, f"{f0_ceil:.1f}", color="r")

    plt.subplot2grid((2, 2), (1, 0))
    _ = plt.hist(
        f0,
        bins=np.logspace(np.log10(f0_floor), np.log10(f0_ceil), 100),
        weights=norm_volume,
    )
    plt.xscale("log")
    plt.title("f0 (volume weighted)")
    plt.axvline(f0_floor, color="r", linestyle="--")
    plt.text(f0_floor, 1, f"{f0_floor:.1f}", color="r")
    plt.axvline(f0_ceil, color="r", linestyle="--")
    plt.text(f0_ceil, 1, f"{f0_ceil:.1f}", color="r")

    plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    plt.hist2d(
        f0,
        volume,
        bins=[np.logspace(np.log10(f0_floor), np.log10(f0_ceil), 100), 100],
        norm="symlog",
    )
    plt.xscale("log")
    plt.colorbar()
    plt.title("f0 vs volume")

    plt.tight_layout()
    filename = verbose_dir / f"calc_lf0_statistics-{i_loop:02d}.png"
    plt.savefig(filename, transparent=False)

    plt.close()


def _compute_final_statistics(f0: np.ndarray) -> LogF0Statistics:
    """最終統計値の計算"""
    lf0 = np.log(f0)
    return LogF0Statistics(
        mean=float(lf0.mean()),
        median=float(np.median(lf0)),
        std=float(np.std(lf0)),
        min=float(lf0.min()),
        max=float(lf0.max()),
        q001=float(np.percentile(lf0, 0.1)),
        q003=float(np.percentile(lf0, 0.3)),
        q005=float(np.percentile(lf0, 0.5)),
        q01=float(np.percentile(lf0, 1)),
        q99=float(np.percentile(lf0, 99)),
        q995=float(np.percentile(lf0, 99.5)),
        q997=float(np.percentile(lf0, 99.7)),
        q999=float(np.percentile(lf0, 99.9)),
    )


def calc_lf0_statistics(
    input_paths: list[Path],
    sampling_rate: float | None,
    max_num: int,
    num_loop: int,
    target_duration: int,
    verbose_dir: Path | None,
) -> LogF0Statistics:
    """
    音声ファイルから対数f0の統計量を計算する。
    f0範囲はループして絞り込む。
    """
    sampling_rate = _validate_sampling_rate(input_paths, sampling_rate)
    paths = _subsample_paths(input_paths, max_num)

    # f0範囲をループして絞り込む
    f0: np.ndarray | None = None
    f0_floor = 40.0  # WORLDのデフォルトは71.0
    f0_ceil = 800.0
    for i_loop in (it := tqdm(range(num_loop))):
        it.set_description(
            f"calc_lf0_statistics: floor: {f0_floor:.2f}Hz, ceil: {f0_ceil:.2f}Hz"
        )

        # f0と音量を計算
        process = partial(
            _calc_features,
            sampling_rate=sampling_rate,
            f0_floor=f0_floor,
            f0_ceil=f0_ceil,
        )
        paths_list = _split_wave_paths(paths, target_duration)
        with Pool() as p:
            features = list(p.imap_unordered(process, paths_list))
            f0s, volumes = zip(*features, strict=True)

        f0 = np.concatenate(f0s)
        volume = np.concatenate(volumes)

        # F0データのフィルタリング
        f0, volume, norm_volume = _filter_f0_data(f0, volume, f0_floor, f0_ceil)

        # 分位数でf0範囲を仮決め
        f0_001 = _weighted_percentile(f0, norm_volume, 0.1)
        f0_999 = _weighted_percentile(f0, norm_volume, 99.9)

        # 対数領域で10%広げたものをfloorとceilにする
        next_f0_floor, next_f0_ceil = _expand_in_logspace(f0_001, f0_999, 0.1)

        # 可視化
        if verbose_dir is not None:
            verbose_dir.mkdir(parents=True, exist_ok=True)
            _create_visualization(
                f0, volume, norm_volume, f0_floor, f0_ceil, verbose_dir, i_loop
            )

        # 更新
        f0_floor = next_f0_floor
        f0_ceil = next_f0_ceil

    assert f0 is not None
    return _compute_final_statistics(f0)


def extract_f0_range(
    input_glob: str,
    output: Path,
    sampling_rate: int | None,
    max_num: int,
    num_loop: int,
    target_duration: int,
    verbose_dir: Path | None,
):
    input_paths = list(map(Path, glob(input_glob)))
    stats = calc_lf0_statistics(
        input_paths=input_paths,
        sampling_rate=sampling_rate,
        max_num=max_num,
        num_loop=num_loop,
        target_duration=target_duration,
        verbose_dir=verbose_dir,
    )

    output_dict = dataclasses.asdict(stats)
    output.write_text(json.dumps(output_dict, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="音声ファイルから適切なF0範囲を自動計算します。反復的にF0を分析し、最適なfloor/ceil値を推定します。統計情報をJSON形式で出力します。"
    )
    parser.add_argument(
        "--input_glob",
        "-i",
        required=True,
        help="入力音声ファイルのパスパターン（例：'*.wav'）",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="F0範囲統計を保存するファイルパス（.json形式）",
    )
    parser.add_argument(
        "--sampling_rate",
        "-sr",
        type=int,
        default=None,
        help="サンプリングレート（Hz）。指定しない場合は自動で検出",
    )
    parser.add_argument(
        "--max_num",
        type=int,
        default=100,
        help="処理する最大ファイル数。大量のファイルがある場合の間引き処理（デフォルト：100）",
    )
    parser.add_argument(
        "--num_loop",
        type=int,
        default=10,
        help="F0範囲推定の反復回数。多いほど精度が向上（デフォルト：10）",
    )
    parser.add_argument(
        "--target_duration",
        type=int,
        default=180,
        help="各反復で処理する目標時間（秒）。処理時間の調整に使用（デフォルト：180秒）",
    )
    parser.add_argument(
        "--verbose_dir",
        type=Path,
        default=None,
        help="詳細な可視化結果を保存するディレクトリ。指定するとヒストグラムが生成されます",
    )
    extract_f0_range(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
