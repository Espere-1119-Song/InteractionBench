"""Test methods."""

from __future__ import annotations

from ..registry import Registry, import_object
from .base import Protocol, ProtocolConfig, make_poll, user_turn  # noqa: F401
from .offline import OfflineProtocol
from .polling import (CumulativeProtocol, InterleavedProtocol, PollingProtocol,  # noqa: F401
                      SlidingProtocol, poll_ticks, trim_context_images)

PROTOCOLS: Registry[type] = Registry("protocol")
for _cls in (SlidingProtocol, CumulativeProtocol, InterleavedProtocol, OfflineProtocol):
    PROTOCOLS.register(_cls.name, _cls)


def register_protocol(name: str, cls: type | None = None, *, overwrite: bool = False):
    return PROTOCOLS.register(name, cls, overwrite=overwrite)


def list_protocols() -> list[str]:
    return PROTOCOLS.names()


def get_protocol(name: str) -> Protocol:
    if name in PROTOCOLS:
        return PROTOCOLS.get(name)()
    if ":" in name:
        return import_object(name)()
    raise KeyError(f"unknown protocol '{name}'. known: {PROTOCOLS.names()} or 'module:Class'")
