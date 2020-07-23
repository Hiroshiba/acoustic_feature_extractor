from pathlib import Path

from setuptools import find_namespace_packages, setup

# console scripts
console_scripts = []

extractors = list(Path("extractor").glob("extract_*"))
assert len(extractors) > 0
console_scripts += [
    f'{path.stem.replace("extract_", "acoustic_feature_extract_")}=extractor.{path.stem}:main'
    for path in extractors
]

analyzers = list(Path("analyzer").glob("analyze_*"))
assert len(analyzers) > 0
console_scripts += [
    f'{path.stem.replace("analyze_", "acoustic_feature_analyze_")}=analyzer.{path.stem}:main'
    for path in analyzers
]

# setup
setup(
    name="acoustic_feature_extractor",
    version="0.0.1",
    packages=find_namespace_packages(),
    url="https://github.com/Hiroshiba/acoustic_feature_extractor",
    author="Kazuyuki Hiroshiba",
    author_email="hihokaruta@gmail.com",
    license="MIT License",
    entry_points=dict(console_scripts=console_scripts),
    install_requires=["numpy", "scipy", "librosa<0.8.0", "pyworld", "pysptk", "tqdm"],
)
