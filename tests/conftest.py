from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def data_dir():
    return Path('/tmp/test_data_acoustic_feature_extractor/')
