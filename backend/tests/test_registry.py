import tempfile

import pytest

from cubist.engines.registry import load_engine


def test_loads_baseline_by_dotted_path():
    eng = load_engine("cubist.engines.baseline")
    assert eng.name == "baseline-v0"


def test_loads_random_by_dotted_path():
    eng = load_engine("cubist.engines.random_engine")
    assert eng.name == "random"


def test_loads_from_file():
    src = """
from cubist.engines.random_engine import RandomEngine
engine = RandomEngine()
"""
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
        f.write(src)
        f.flush()
        eng = load_engine(f.name)
    assert eng.name == "random"


def test_rejects_module_without_engine_symbol(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("x = 1")
    with pytest.raises(AttributeError):
        load_engine(str(bad))
