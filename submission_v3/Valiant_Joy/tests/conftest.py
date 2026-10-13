import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: runs real scripts with --quick (downloads data once)")


@pytest.fixture(scope="session")
def dummy():
    from dummy_adapter import DummyAdapter

    return DummyAdapter()


@pytest.fixture(scope="session")
def trained(dummy):
    return dummy.finetune(dummy.build_model(), 3)
