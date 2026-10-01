"""Offline tests for the one-shot `laya-mlx-decide` CLI.

No checkpoints are loaded: a stub Agent replaces the model, so these cover
argument handling, validation, exit codes and output formatting only.
"""

import json

import pytest

from laya_mlx.service import decide

QUESTION = '{"a": {"type": "noul", "instructions": "Is a refund requested?"}}'


class StubAgent:
    """Answers without MLX so the CLI can be exercised end to end."""

    last = {}

    def __init__(self, path, **kwargs):
        StubAgent.last = {"path": path, **kwargs}

    def predict(self, state, questions):
        return {
            "model": "laya-rl-agent",
            "answers": {
                qid: {
                    "type": d["type"],
                    "noul": 0.5,
                    "confidence": 0.5,
                    "score": 1.0,
                    "choice": list(d.get("criteria", ["x"]))[0],
                    "probabilities": {"x": 1.0},
                    "legend": {"0": "low", "1": "high"},
                }
                for qid, d in questions.items()
            },
            "usage": {"input_tokens": 1, "output_tokens": 0},
        }


@pytest.fixture
def stub_model(tmp_path, monkeypatch):
    """Point --model at a real directory and swap in StubAgent."""
    path = tmp_path / "ckpt"
    (path / "encoder").mkdir(parents=True)
    (path / "tokenizer").mkdir()
    (path / "encoder" / "config.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"x")
    (path / "rl_agent_config.json").write_text("{}")
    import laya_mlx.agent as agent_mod

    monkeypatch.setattr(agent_mod, "Agent", StubAgent)
    return path


# --------------------------------------------------------------------- formatting
def test_format_answers_defaults_to_json():
    answers = {
        "a": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.9, "tech": 0.1}}
    }
    out = json.loads(decide.format_answers(answers))
    assert out == answers


def test_format_answers_compact_renders_each_type():
    answers = {
        "team": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.9, "tech": 0.1},
        },
        "level": {"type": "score", "score": 1.5, "legend": {"0": "a", "1": "b"}},
        "urgent": {"type": "noul", "noul": 0.25},
    }
    lines = decide.format_answers(answers, compact=True).splitlines()
    assert lines[0].startswith("team: billing (")
    assert lines[1] == "level: 1.500 of 2"
    assert lines[2] == "urgent: 0.2500"


# ------------------------------------------------------------------------- usage
def test_missing_model_exits_with_usage_code():
    """argparse exits 2 on a missing required argument, which is EXIT_USAGE."""
    with pytest.raises(SystemExit) as exit_info:
        decide.main([])
    assert exit_info.value.code == decide.EXIT_USAGE


def test_unknown_preset_is_rejected():
    with pytest.raises(SystemExit):
        decide.parse_args(["--model", "/tmp/x", "--preset", "nope"])


def test_preset_cannot_be_combined_with_questions():
    args = decide.parse_args(["--model", "/tmp/x", "--preset", "triage", "--questions", QUESTION])
    with pytest.raises(SystemExit, match="on its own"):
        decide._read_questions(args)


def test_questions_and_questions_file_are_exclusive():
    args = decide.parse_args(
        ["--model", "/tmp/x", "--questions", QUESTION, "--questions-file", "q.json"]
    )
    with pytest.raises(SystemExit, match="only one"):
        decide._read_questions(args)


def test_no_questions_source_is_an_error():
    args = decide.parse_args(["--model", "/tmp/x"])
    with pytest.raises(SystemExit, match="Provide one of"):
        decide._read_questions(args)


def test_invalid_question_type_is_rejected():
    args = decide.parse_args(
        ["--model", "/tmp/x", "--questions", '{"a": {"type": "bogus", "instructions": "x"}}']
    )
    with pytest.raises(Exception, match="unknown type"):
        decide._read_questions(args)


def test_duplicate_choice_labels_are_rejected():
    args = decide.parse_args(
        [
            "--model",
            "/tmp/x",
            "--questions",
            '{"a": {"type": "choice", "instructions": "x", "criteria": ["p", "p"]}}',
        ]
    )
    with pytest.raises(Exception, match="duplicate"):
        decide._read_questions(args)


def test_preset_expands_to_questions():
    args = decide.parse_args(["--model", "/tmp/x", "--preset", "triage"])
    assert "intent" in decide._read_questions(args)


def test_questions_file_is_read(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(QUESTION)
    args = decide.parse_args(["--model", "/tmp/x", "--questions-file", str(path)])
    assert "a" in decide._read_questions(args)


def test_state_from_flag():
    args = decide.parse_args(["--model", "/tmp/x", "--state", "hello"])
    assert decide._read_state(args) == "hello"


def test_state_and_state_file_are_exclusive():
    args = decide.parse_args(["--model", "/tmp/x", "--state", "x", "--state-file", "s.json"])
    with pytest.raises(SystemExit, match="only one"):
        decide._read_state(args)


# ------------------------------------------------------------------------- paths
def test_missing_checkpoint_exits_with_model_code(capsys, tmp_path):
    code = decide.main(["--model", str(tmp_path / "nope"), "--state", "x", "--questions", QUESTION])
    assert code == decide.EXIT_MODEL
    assert "does not exist" in capsys.readouterr().err


def test_describe_prints_metadata_without_loading(capsys, stub_model):
    code = decide.main(["--model", str(stub_model), "--describe"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["layout"] == "hub"
    assert StubAgent.last == {}, "--describe must not construct the Agent"


def test_flat_checkpoint_is_linked(stub_model, monkeypatch, capsys):

    flat = stub_model.with_name("flat")
    flat.mkdir()
    (flat / "config.json").write_text("{}")
    (flat / "model.safetensors").write_bytes(b"x")
    (flat / "rl_agent_config.json").write_text("{}")
    (flat / "tokenizer.json").write_text("{}")
    assert decide.main(["--model", str(flat), "--state", "x", "--questions", QUESTION]) == 0
    capsys.readouterr()
    assert (flat.with_name("flat_hub") / "encoder" / "config.json").is_file()


# ------------------------------------------------------------------------ running
def test_predict_prints_answers(stub_model, capsys):
    code = decide.main(["--model", str(stub_model), "--state", "x", "--questions", QUESTION])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["a"]["type"] == "noul"


def test_predict_passes_flags_through(stub_model, capsys):
    decide.main(
        [
            "--model",
            str(stub_model),
            "--state",
            "x",
            "--questions",
            QUESTION,
            "--dtype",
            "float32",
            "--batch-size",
            "8",
        ]
    )
    capsys.readouterr()
    assert StubAgent.last["dtype"] == "float32"
    assert StubAgent.last["batch_size"] == 8


def test_model_value_error_exits_with_model_code(stub_model, monkeypatch, capsys):
    class Boom(StubAgent):
        def predict(self, state, questions):
            raise ValueError("non-finite model outputs")

    import laya_mlx.agent as agent_mod

    monkeypatch.setattr(agent_mod, "Agent", Boom)
    code = decide.main(["--model", str(stub_model), "--state", "x", "--questions", QUESTION])
    assert code == decide.EXIT_MODEL
    assert "non-finite" in capsys.readouterr().err
