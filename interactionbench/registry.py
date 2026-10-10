"""A minimal name -> object registry shared by models, judges and protocols."""

from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str, obj: T | None = None, *, overwrite: bool = False):
        def _add(o: T) -> T:
            if name in self._items and not overwrite:
                raise ValueError(f"{self.kind} '{name}' is already registered")
            self._items[name] = o
            return o
        return _add if obj is None else _add(obj)

    def get(self, name: str) -> T:
        if name not in self._items:
            raise KeyError(f"unknown {self.kind} '{name}'. known: {sorted(self._items)}")
        return self._items[name]

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def names(self) -> list[str]:
        return sorted(self._items)


def import_object(path: str):
    import importlib
    import importlib.util
    from pathlib import Path

    mod_name, _, attr = path.partition(":")
    if not attr:
        raise ValueError(f"expected 'module:attribute', got {path!r}")
    if mod_name.endswith(".py") or "/" in mod_name:
        fp = Path(mod_name)
        spec = importlib.util.spec_from_file_location(fp.stem, fp)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import {fp}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(mod_name)
    return getattr(module, attr)


_LOADED_PLUGINS: set[str] = set()


def load_plugins(paths: list[str] | None) -> None:
    import importlib
    import importlib.util
    import sys
    from pathlib import Path

    for p in paths or []:
        if p.endswith(".py") or "/" in p:
            fp = Path(p).resolve()
            if str(fp) in _LOADED_PLUGINS:
                continue
            _LOADED_PLUGINS.add(str(fp))
            spec = importlib.util.spec_from_file_location(f"ibench_plugin_{fp.stem}", fp)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot import plugin {fp}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        else:
            importlib.import_module(p)
