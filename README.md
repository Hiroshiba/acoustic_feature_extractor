# acoustic_feature_extractor

Python 3.11

## 使い方

### uvx

```bash
# 例：メルスペクトログラムを抽出
uvx \
  --from git+https://github.com/Hiroshiba/acoustic_feature_extractor \
  acoustic_feature_extract_melspectrogram \
  --input_glob "/path/to/dir/*" \
  --output_directory /path/to/dir
```

### PyPI

```bash
pip install git+https://github.com/Hiroshiba/acoustic_feature_extractor

# 例：メルスペクトログラムを抽出
acoustic_feature_extract_melspectrogram \
  --input_glob "/path/to/dir/*" \
  --output_directory /path/to/dir
```

## テスト

```bash
uv run pytest -sv tests/
```

### スナップショットの更新

```bash
uv run pytest --snapshot-update
```

## 静的解析・フォーマット

```bash
uv run ruff check --fix && uv run ruff format
```
