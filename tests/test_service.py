"""Offline tests for the HTTP service layer.

No checkpoints and no MLX weights are loaded: routing runs against a stub
Registry, the HTTP handlers against a stub Agent. What is verified here is the
service contract -- status codes, payload shape, routing rule, and the
in-flight bound -- not model accuracy.
"""

import json
import threading
import urllib.error
import urllib.request

import pytest

from laya_mlx.service.checkpoint import CheckpointError, link_flat_checkpoint, resolve_checkpoint
from laya_mlx.service.registry import Registry

WEIGHTS = b"not-really-safetensors"


def make_flat_checkpoint(root, name="flat"):
    """A directory shaped like an LM Studio download: flat, no encoder/ subdir.

    LM Studio flattens the Hub layout, so `config.json` and the tokenizer files
    sit at the top level rather than under `encoder/` and `tokenizer/`.
    """
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text('{"model_type": "modernbert"}')
    (path / "model.safetensors").write_bytes(WEIGHTS)
    (path / "rl_agent_config.json").write_text('{"encoder": {}, "head_layers": []}')
    (path / "mlx_config.json").write_text('{"format": "laya-mlx", "dtype": "float16"}')
    (path / "tokenizer.json").write_text("{}")
    (path / "tokenizer_config.json").write_text("{}")
    return path


def make_hub_checkpoint(root, name="hub"):
    """A directory already in Hub layout, as `laya.load` expects."""
    path = root / name
    (path / "encoder").mkdir(parents=True)
    (path / "tokenizer").mkdir(parents=True)
    (path / "encoder" / "config.json").write_text('{"model_type": "modernbert"}')
    (path / "model.safetensors").write_bytes(WEIGHTS)
    (path / "rl_agent_config.json").write_text('{"encoder": {}, "head_layers": []}')
    (path / "tokenizer" / "tokenizer.json").write_text("{}")
    return path


# --------------------------------------------------------------------- checkpoint
def test_link_flat_checkpoint_creates_hub_layout(tmp_path):
    source = make_flat_checkpoint(tmp_path)
    output = link_flat_checkpoint(source, tmp_path / "linked")
    assert (output / "encoder" / "config.json").is_file()
    assert (output / "tokenizer" / "tokenizer.json").is_file()
    assert (output / "model.safetensors").is_file()


def test_link_uses_symlinks_so_weights_are_not_copied(tmp_path):
    source = make_flat_checkpoint(tmp_path)
    output = link_flat_checkpoint(source, tmp_path / "linked")
    assert (output / "model.safetensors").is_symlink()
    assert (output / "model.safetensors").stat().st_size == len(WEIGHTS)


def test_link_refuses_to_overwrite(tmp_path):
    source = make_flat_checkpoint(tmp_path)
    link_flat_checkpoint(source, tmp_path / "linked")
    with pytest.raises(CheckpointError, match="already exists"):
        link_flat_checkpoint(source, tmp_path / "linked")


def test_link_rejects_source_already_in_hub_layout(tmp_path):
    source = make_hub_checkpoint(tmp_path)
    with pytest.raises(CheckpointError, match="already in Hub layout"):
        link_flat_checkpoint(source, tmp_path / "out")


def test_resolve_returns_hub_layout_unchanged(tmp_path):
    source = make_flat_checkpoint(tmp_path)
    linked = link_flat_checkpoint(source, tmp_path / "linked")
    assert resolve_checkpoint(linked) == linked.resolve()


def test_resolve_links_a_flat_checkpoint_once(tmp_path):
    source = make_flat_checkpoint(tmp_path)
    first = resolve_checkpoint(source)
    second = resolve_checkpoint(source)
    assert first == second


def test_resolve_rejects_a_non_checkpoint_directory(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(CheckpointError, match="Not a Laya checkpoint"):
        resolve_checkpoint(tmp_path / "empty")


def test_resolve_rejects_a_missing_directory(tmp_path):
    with pytest.raises(CheckpointError, match="does not exist"):
        resolve_checkpoint(tmp_path / "nope")


# ------------------------------------------------------------------------ routing
def test_choose_routes_non_latin_to_multilingual(tmp_path):
    registry = Registry(
        {
            "english": make_flat_checkpoint(tmp_path, "en"),
            "multilingual": make_flat_checkpoint(tmp_path, "multi"),
        }
    )
    name, reason = registry.choose("发票被重复扣款")
    assert name == "multilingual"
    assert "non-Latin script" in reason


def test_choose_routes_english_to_english(tmp_path):
    registry = Registry(
        {
            "english": make_flat_checkpoint(tmp_path, "en"),
            "multilingual": make_flat_checkpoint(tmp_path, "multi"),
        }
    )
    name, _ = registry.choose("I was double charged and want a refund")
    assert name == "english"


def test_choose_falls_back_when_only_one_checkpoint(tmp_path):
    registry = Registry({"multilingual": make_flat_checkpoint(tmp_path, "multi")})
    name, reason = registry.choose("I was double charged")
    assert name == "multilingual"
    assert "fallback" in reason


def test_choose_handles_text_without_letters(tmp_path):
    registry = Registry({"english": make_flat_checkpoint(tmp_path, "en")})
    name, reason = registry.choose("12345 !!! ???")
    assert name == "english"
    assert "no letters" in reason


# ----------------------------------------------------------------------- registry
def test_registry_rejects_an_unknown_checkpoint(tmp_path):
    registry = Registry({"english": make_flat_checkpoint(tmp_path, "en")})
    with pytest.raises(CheckpointError, match="Unknown checkpoint"):
        registry.get("nope")


def test_inflight_semaphore_bounds_concurrency(tmp_path):
    registry = Registry({}, max_inflight=1)
    assert registry.acquire_slot() is True
    assert registry.acquire_slot(timeout=0.05) is False
    registry.release_slot()
    assert registry.acquire_slot(timeout=0.05) is True


def test_inflight_is_released_under_threads(tmp_path):
    registry = Registry({}, max_inflight=4)
    errors = []

    def worker():
        try:
            for _ in range(50):
                if not registry.acquire_slot(timeout=5):
                    errors.append("could not acquire")
                    return
                registry.release_slot()
        except Exception as error:  # pragma: no cover - surfaced via errors list
            errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors


# ----------------------------------------------------------------------- HTTP API
class StubAgent:
    """Answers without touching MLX: one noul, or echoes the questions back."""

    def __init__(self, name):
        self.name = name

    def predict(self, state, questions):
        answers = {}
        for qid, definition in questions.items():
            kind = definition["type"]
            if kind == "choice":
                labels = list(definition["criteria"])
                answers[qid] = {
                    "type": kind,
                    "choice": labels[0],
                    "confidence": 1.0,
                    "probabilities": dict.fromkeys(labels, 1.0 / len(labels)),
                }
            elif kind == "score":
                answers[qid] = {"type": kind, "score": 0.0, "confidence": 1.0}
            else:
                answers[qid] = {"type": kind, "noul": 0.25, "confidence": 0.25}
        return {
            "model": "laya-rl-agent",
            "answers": answers,
            "usage": {"input_tokens": 1, "output_tokens": 0},
        }


@pytest.fixture
def service(tmp_path):
    registry = Registry(
        {
            "english": make_flat_checkpoint(tmp_path, "en"),
            "multilingual": make_flat_checkpoint(tmp_path, "multi"),
        }
    )
    registry._agents = {"english": StubAgent("english"), "multilingual": StubAgent("multilingual")}
    from laya_mlx.service.http import Handler, Server

    handler = type("BoundHandler", (Handler,), {"registry": registry})
    server = Server(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield "http://127.0.0.1:%d" % server.server_address[1]
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def call(base, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8", "replace"))


def test_health_lists_checkpoint_metadata(service):
    status, body = call(service, "/health")
    assert status == 200
    assert body["status"] == "ok"
    assert set(body["checkpoints"]) == {"english", "multilingual"}


def test_presets_endpoint_lists_bundled_presets(service):
    status, body = call(service, "/presets")
    assert status == 200
    assert "triage" in body["presets"]


def test_predict_expands_a_preset(service):
    status, body = call(service, "/predict", {"state": "hello", "preset": "triage"})
    assert status == 200
    assert set(body["answers"]) == set(
        __import__("laya_mlx.service.http", fromlist=["PRESETS"]).PRESETS["triage"]()
    )


def test_predict_routes_when_no_checkpoint_named(service):
    status, body = call(
        service,
        "/predict",
        {"state": "发票被重复扣款", "questions": {"a": {"type": "noul", "instructions": "x"}}},
    )
    assert status == 200
    assert body["checkpoint"] == "multilingual"


def test_route_reports_reason_and_evidence(service):
    status, body = call(
        service,
        "/route",
        {"state": "I want a refund", "questions": {"a": {"type": "noul", "instructions": "x"}}},
    )
    assert status == 200
    assert body["checkpoint"] == "english"
    assert body["routing_reason"]
    assert body["language"]["script"] == "latin"


def test_predict_rejects_missing_state(service):
    status, body = call(
        service, "/predict", {"questions": {"a": {"type": "noul", "instructions": "x"}}}
    )
    assert status == 400
    assert "state" in body["error"]


def test_predict_rejects_unknown_question_type(service):
    status, _ = call(
        service,
        "/predict",
        {"state": "x", "questions": {"a": {"type": "bogus", "instructions": "x"}}},
    )
    assert status == 400


def test_predict_rejects_empty_questions(service):
    status, _ = call(service, "/predict", {"state": "x", "questions": {}})
    assert status == 400


def test_predict_rejects_unknown_checkpoint(service):
    status, body = call(
        service,
        "/predict",
        {
            "state": "x",
            "checkpoint": "nope",
            "questions": {"a": {"type": "noul", "instructions": "x"}},
        },
    )
    assert status == 503
    assert "Unknown checkpoint" in body["error"]


def test_malformed_json_is_a_400(service):
    request = urllib.request.Request(
        service + "/predict", data=b"{not json", headers={"Content-Type": "application/json"}
    )
    try:
        urllib.request.urlopen(request, timeout=30)
        raise AssertionError("expected HTTPError")
    except urllib.error.HTTPError as error:
        assert error.code == 400


def test_unknown_route_is_a_404(service):
    status, _ = call(service, "/nope", {})
    assert status == 404


def test_shortlist_requires_labels(service):
    status, _ = call(service, "/shortlist", {"state": "x"})
    assert status == 400


def test_shortlist_rejects_non_string_labels(service):
    status, _ = call(service, "/shortlist", {"state": "x", "labels": [1, 2, 3]})
    assert status == 400
