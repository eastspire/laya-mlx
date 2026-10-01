"""Serve local Laya checkpoints over HTTP for other models and agents."""

import argparse
from pathlib import Path

from .. import __version__
from .checkpoint import CheckpointError
from .http import build_server


def _split(value):
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="laya-mlx-serve", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: %(default)s)")
    parser.add_argument("--port", type=int, default=8080, help="Bind port (default: %(default)s)")
    parser.add_argument(
        "--checkpoint",
        action="append",
        metavar="NAME=PATH",
        help="Register a checkpoint, repeatable. Default: english=..., multilingual=...",
    )
    parser.add_argument(
        "--dtype", choices=("float16", "float32"), default="float16", help="Compute precision"
    )
    parser.add_argument("--no-compile", action="store_true", help="Disable mx.compile")
    parser.add_argument(
        "--max-inflight",
        type=int,
        default=16,
        help="Concurrent inference slots (default: %(default)s)",
    )
    parser.add_argument(
        "--queue-timeout",
        type=float,
        default=60.0,
        help="Seconds a request waits for a slot before 503 (default: %(default)s)",
    )
    parser.add_argument(
        "--preload",
        default="",
        help="Comma-separated checkpoint names to load at startup",
    )
    return parser.parse_args(argv)


def resolve_checkpoints(values):
    """Turn NAME=PATH arguments (or the defaults) into a name -> path mapping."""
    if not values:
        from ..router import DEFAULT_MODELS

        return dict(DEFAULT_MODELS)
    checkpoints = {}
    for item in values:
        if "=" not in item:
            raise SystemExit("--checkpoint expects NAME=PATH, got %r" % item)
        name, path = item.split("=", 1)
        name = name.strip()
        if not name or not path.strip():
            raise SystemExit("--checkpoint expects NAME=PATH, got %r" % item)
        if name in checkpoints:
            raise SystemExit("Duplicate checkpoint name %r" % name)
        checkpoints[name] = Path(path).expanduser()
    return checkpoints


def main(argv=None):
    args = parse_args(argv)
    try:
        server = build_server(
            args.host,
            args.port,
            checkpoints=resolve_checkpoints(args.checkpoint),
            dtype=args.dtype,
            compile=not args.no_compile,
            max_inflight=args.max_inflight,
            queue_timeout=args.queue_timeout,
        )
    except CheckpointError as error:
        raise SystemExit(str(error)) from error

    registry = server.RequestHandlerClass.registry
    for name in _split(args.preload):
        try:
            registry.get(name)
        except CheckpointError as error:
            raise SystemExit(str(error)) from error

    print("[serve] laya-mlx %s on http://%s:%d" % (__version__, args.host, args.port), flush=True)
    for name, path in sorted(registry.checkpoints.items()):
        print("[serve]   %s -> %s" % (name, path), flush=True)
    print(
        "[serve] max inflight: %d, queue timeout: %gs" % (args.max_inflight, args.queue_timeout),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] stopped", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
