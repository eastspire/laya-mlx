"""HTTP service for local Laya checkpoints.

Exposes typed-decision inference over JSON so other models, scripts or agents
can call a local checkpoint without importing `laya_mlx` or knowing where the
weights live.

    laya-mlx-serve --checkpoint english=./models/laya \\
                   --checkpoint multilingual=./models/laya-multilingual
"""

from .checkpoint import CheckpointError, describe, link_flat_checkpoint, resolve_checkpoint
from .http import PRESETS, BusyError, Handler, Server, ServiceError, build_server
from .registry import ENGLISH, MULTILINGUAL, Registry

__all__ = [
    "BusyError",
    "CheckpointError",
    "ENGLISH",
    "Handler",
    "MULTILINGUAL",
    "PRESETS",
    "Registry",
    "Server",
    "ServiceError",
    "build_server",
    "describe",
    "link_flat_checkpoint",
    "resolve_checkpoint",
]
