from pathlib import Path

from setuptools import setup, find_packages

# console scripts
extractors = list(Path('extractor').glob('extract_*'))
assert len(extractors) > 0

console_scripts = [
    f'{path.stem.replace("extract_", "acoustic_feature_extract_")}=extractor.{path.stem}:main'
    for path in extractors
]

setup(
    name='acoustic_feature_extractor',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/Hiroshiba/acoustic_feature_extractor',
    author='Kazuyuki Hiroshiba',
    author_email='hihokaruta@gmail.com',
    license='MIT License',
    entry_points=dict(console_scripts=console_scripts),
    install_requires=[
        'numpy',
        'scipy',
        'librosa<0.8.0',
        'pyworld',
        'pysptk',
        'tqdm',
        'kiritan_singing_label_reader @ git+https://github.com/Hiroshiba/kiritan_singing_label_reader',
    ],
)
