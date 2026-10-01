"""One-shot typed decisions from the command line, without a resident server.

Loading the weights is the expensive part of every call, so a shell pipeline that
invokes this repeatedly is slower than a resident service. What this buys is a
process that exits, holds no port, and can be called from cron, a Makefile, or
another program's `subprocess`.

    laya-mlx-decide --model ./models/laya-multilingual \\
                    --state '发票被重复扣款' --preset triage

Prints the answers dict as JSON on stdout. Exits non-zero on a bad request, so
`set -e` in a shell script does the right thing.
"""

import argparse
import json
import sys
from pathlib import Path

from .checkpoint import CheckpointError, describe, resolve_checkpoint
from .http import PRESETS, ServiceError, validate_question

EXIT_USAGE = 2
EXIT_MODEL = 3


def _read_questions(args):
    """Resolve --questions / --questions-file / --preset into one dict."""
    if args.questions is not None and args.questions_file is not None:
        raise SystemExit("Use only one of --questions or --questions-file")
    if args.preset and (args.questions or args.questions_file):
        raise SystemExit("Use --preset on its own, not with --questions")

    if args.preset:
        factory = PRESETS.get(args.preset)
        if factory is None:
            raise SystemExit(
                "Unknown preset %r; available: %s" % (args.preset, ", ".join(sorted(PRESETS)))
            )
        questions = factory()
    elif args.questions is not None:
        questions = json.loads(args.questions)
    elif args.questions_file is not None:
        questions = json.loads(Path(args.questions_file).read_text(encoding="utf-8"))
    else:
        raise SystemExit("Provide one of --questions, --questions-file or --preset")

    if not isinstance(questions, dict) or not questions:
        raise SystemExit("Questions must be a nonempty JSON object keyed by question id")
    for qid, definition in questions.items():
        validate_question(qid, definition)
    return questions


def _read_state(args):
    if args.state is not None and args.state_file is not None:
        raise SystemExit("Use only one of --state or --state-file")
    if args.state is not None:
        return args.state
    if args.state_file is not None:
        return json.loads(Path(args.state_file).read_text(encoding="utf-8"))
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("Provide --state, --state-file, or pipe the state on stdin")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="laya-mlx-decide",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Checkpoint directory or Hub id")
    parser.add_argument("--state", help="State text")
    parser.add_argument("--state-file", type=Path, help="JSON state file")
    parser.add_argument("--questions", help="Inline JSON question definitions")
    parser.add_argument("--questions-file", type=Path, help="JSON question file")
    parser.add_argument("--preset", choices=sorted(PRESETS), help="Use a bundled question preset")
    parser.add_argument(
        "--dtype", choices=("float16", "float32"), default="float16", help="Compute precision"
    )
    parser.add_argument("--device", choices=("gpu", "cpu"), help="MLX device")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--compact", action="store_true", help="Emit one answer per line")
    parser.add_argument(
        "--describe", action="store_true", help="Print checkpoint metadata and exit"
    )
    return parser.parse_args(argv)


def format_answers(answers, compact=False):
    """Render answers for a human, or as JSON when compact is not set."""
    if not compact:
        return json.dumps(answers, indent=2, ensure_ascii=False)
    lines = []
    for qid, answer in answers.items():
        if answer["type"] == "choice":
            value = "%s (%s)" % (answer["choice"], answer["probabilities"])
        elif answer["type"] == "score":
            value = "%.3f of %d" % (answer["score"], len(answer.get("legend", {})))
        else:
            value = "%.4f" % answer["noul"]
        lines.append("%s: %s" % (qid, value))
    return "\n".join(lines)


def main(argv=None):
    args = parse_args(argv)
    try:
        path = resolve_checkpoint(args.model)
    except CheckpointError as error:
        print("laya-mlx-decide: %s" % error, file=sys.stderr)
        return EXIT_MODEL

    if args.describe:
        print(json.dumps(describe(path), indent=2))
        return 0

    try:
        questions = _read_questions(args)
        state = _read_state(args)
    except SystemExit as error:
        print("laya-mlx-decide: %s" % error, file=sys.stderr)
        return EXIT_USAGE
    except (ServiceError, json.JSONDecodeError, OSError) as error:
        print("laya-mlx-decide: %s" % error, file=sys.stderr)
        return EXIT_USAGE

    # Imported here so --describe and usage errors never pay for MLX startup.
    from ..agent import Agent

    try:
        agent = Agent(
            str(path),
            dtype=args.dtype,
            device=args.device,
            batch_size=args.batch_size,
        )
        answers = agent.predict(state, questions)["answers"]
    except (ValueError, FloatingPointError) as error:
        print("laya-mlx-decide: %s: %s" % (type(error).__name__, error), file=sys.stderr)
        return EXIT_MODEL

    print(format_answers(answers, compact=args.compact))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
