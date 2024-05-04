# acoustic_feature_extractor

Python 3.11

## Install

```bash
pip install git+https://github.com/Hiroshiba/acoustic_feature_extractor
```

## Usabe

```bash
# ex. extract melspectrogram
acoustic_feature_extract_melspectrogram \
  --input_glob "/path/to/dir/*" \
  --output_directory /path/to/dir
```

## Tests

```bash
pytest -sv tests/
```
