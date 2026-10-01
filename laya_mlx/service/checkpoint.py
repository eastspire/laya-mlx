"""Resolve a local checkpoint directory into the flat/nested layouts in the wild.

`laya.load` expects the Hugging Face layout (`encoder/config.json` plus a
`tokenizer/` subdirectory). Desktop tools do not always preserve it: LM Studio,
for example, flattens the repository into a single directory with a top-level
`config.json` and `tokenizer.json`. Those downloads still contain every file the
runtime needs, so this module re-links them into the expected shape without
copying weights or mutating the original directory.
"""

import json
import os
import shutil
from pathlib import Path

FLAT_ENCODER = "config.json"
NESTED_ENCODER = "encoder/config.json"
NESTED_TOKENIZER = "tokenizer"


class CheckpointError(RuntimeError):
    """Raised when a directory is not a usable Laya checkpoint."""


def _is_hub_layout(path):
    return (path / NESTED_ENCODER).is_file() and (path / NESTED_TOKENIZER).is_dir()


def _is_flat_layout(path):
    if not (path / FLAT_ENCODER).is_file():
        return False
    return any(name.startswith("tokenizer") for name in (child.name for child in path.iterdir()))


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointError(f"Unreadable JSON at {path}: {error}") from error


def link_flat_checkpoint(source, output):
    """Re-link a flattened checkpoint directory into the Hub layout.

    Symlinks are used so the weights are not duplicated on disk. Returns the
    output directory.
    """
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not source.is_dir():
        raise CheckpointError(f"Source directory does not exist: {source}")
    if output.exists():
        raise CheckpointError(f"Output already exists: {output}")
    if _is_hub_layout(source):
        raise CheckpointError(f"Source is already in Hub layout: {source}")

    output.mkdir(parents=True)
    (output / NESTED_ENCODER).parent.mkdir(parents=True, exist_ok=True)
    (output / NESTED_TOKENIZER).mkdir(parents=True, exist_ok=True)

    def link(name, target):
        origin = source / name
        if origin.is_file():
            os.symlink(origin, target)

    link(FLAT_ENCODER, output / NESTED_ENCODER)
    link("model.safetensors", output / "model.safetensors")
    link("rl_agent_config.json", output / "rl_agent_config.json")
    link("mlx_config.json", output / "mlx_config.json")
    for child in sorted(source.iterdir()):
        if child.is_file() and child.name.startswith("tokenizer"):
            link(child.name, output / NESTED_TOKENIZER / child.name)

    missing = [
        name
        for name in ("model.safetensors", "rl_agent_config.json", NESTED_ENCODER)
        if not (output / name).is_file()
    ]
    if missing:
        shutil.rmtree(output, ignore_errors=True)
        raise CheckpointError(f"Source is missing required files: {', '.join(missing)}")
    return output


def resolve_checkpoint(path):
    """Return a directory `laya.load` accepts, linking a flat checkpoint if needed.

    A directory already in Hub layout is returned unchanged. A flattened one is
    linked into a sibling `_hub` directory on first use and reused afterwards.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise CheckpointError(f"Checkpoint directory does not exist: {path}")
    if _is_hub_layout(path):
        return path
    if not _is_flat_layout(path):
        raise CheckpointError(
            f"Not a Laya checkpoint: {path} has neither {NESTED_ENCODER} "
            f"nor a top-level {FLAT_ENCODER}"
        )
    linked = path.with_name(path.name + "_hub")
    if not linked.is_dir():
        link_flat_checkpoint(path, linked)
    return linked


def describe(path):
    """Summarise a checkpoint directory for /health style responses."""
    path = Path(path).expanduser()
    meta = path / "mlx_config.json"
    values = _read_json(meta) if meta.is_file() else {}
    weights = path / "model.safetensors"
    return {
        "path": str(path),
        "layout": "hub"
        if _is_hub_layout(path)
        else ("flat" if _is_flat_layout(path) else "unknown"),
        "dtype": values.get("dtype"),
        "format": values.get("format"),
        "weights_bytes": weights.stat().st_size if weights.is_file() else None,
    }
