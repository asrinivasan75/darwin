"""Dynamic engine loading."""

import importlib
import importlib.util
import sys
from pathlib import Path

from cubist.engines.base import Engine

GENERATED_DIR = Path(__file__).parent / "generated"


def load_engine(module_path: str) -> Engine:
    """Import a module by dotted path or file path; return its `engine` symbol."""
    if module_path.endswith(".py") or "/" in module_path or "\\" in module_path:
        path = Path(module_path)
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {module_path}")
        mod = importlib.util.module_from_spec(spec)
        # Register in sys.modules BEFORE exec_module so:
        #   1. The class's __module__ attribute resolves to a real
        #      sys.modules entry, letting inspect.getsource() find the
        #      source file when the orchestrator later asks for it as
        #      the new gen's primary champion.
        #   2. Any internal `from .x import y` style relative imports
        #      inside the loaded engine module work — though our engines
        #      don't use them, this is the standard importlib idiom.
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(module_path)

    eng = getattr(mod, "engine", None)
    if eng is None:
        raise AttributeError(f"{module_path} has no top-level `engine` symbol")
    if not isinstance(eng, Engine):
        raise TypeError(f"{module_path}.engine does not satisfy Engine Protocol")
    return eng


def list_generated() -> list[Path]:
    return sorted(GENERATED_DIR.glob("*.py"))
