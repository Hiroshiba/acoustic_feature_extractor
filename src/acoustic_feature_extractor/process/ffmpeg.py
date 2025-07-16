import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from acoustic_feature_extractor.data.wave import Wave


@dataclass
class Ebur128:
    """
    ebur128の値
    """

    I: float  # noqa: E741
    LRA_low: float
    LRA_high: float

    def __str__(self):
        return (
            f"I: {self.I:.2f} LUFS,"
            f"LRA_high: {self.LRA_high:.2f} LUFS,"
            f"LRA_low: {self.LRA_low:.2f} LUFS"
        )


def has_ffmpeg():
    """
    ffmpegが存在するかチェック
    """
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        print("ffmpegがありません", file=sys.stderr)
        return False


def calc_ebur128(wave: Wave):
    """
    ffmpegを用いてebur128を計算する
    """
    assert has_ffmpeg()

    with NamedTemporaryFile(suffix=".wav") as f:
        wave.save(Path(f.name))

        cmd = (
            "ffmpeg",
            "-nostats",
            *("-i", f.name),
            *("-filter_complex", "ebur128"),
            *("-f", "null"),
            "-",
        )
        output = subprocess.run(
            cmd, check=True, capture_output=True, stdin=subprocess.DEVNULL
        ).stderr.decode()

    output = output.split("Summary:")[-1]
    return Ebur128(
        I=float(output.split("I:")[1].split("LUFS")[0].strip()),
        LRA_low=float(output.split("LRA low:")[1].split("LUFS")[0].strip()),
        LRA_high=float(output.split("LRA high:")[1].split("LUFS")[0].strip()),
    )
