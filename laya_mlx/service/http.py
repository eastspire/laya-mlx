"""JSON/HTTP surface over local Laya checkpoints.

Endpoints:
    GET  /health     liveness, resident checkpoints, per-checkpoint metadata
    GET  /presets    names of the bundled question presets
    POST /predict    state + questions -> typed answers
    POST /route      same, choosing the checkpoint from the input's script
    POST /shortlist  choice over a large label set, embedding-filtered to top-k
"""

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .. import __version__
from ..presets import (
    email_questions,
    guard_questions,
    moderation_questions,
    router_questions,
    triage_questions,
)
from ..shortlist import embed_fn_from_agent, predict_shortlist
from .checkpoint import CheckpointError, describe
from .registry import Registry

PRESETS = {
    "triage": triage_questions,
    "email": email_questions,
    "guard": guard_questions,
    "moderation": moderation_questions,
    "router": router_questions,
}
MAX_BODY_BYTES = 8 * 1024 * 1024
QUESTION_TYPES = ("choice", "score", "noul")


class BusyError(TimeoutError):
    """Raised when every inference slot is occupied past the queue timeout."""


class ServiceError(ValueError):
    """Raised for a malformed request; reported as 400."""


def validate_question(qid, definition):
    """Reject a malformed question before it reaches the model.

    The Agent raises ValueError for the same conditions, but only after the
    request has already claimed an inference slot. A request body is untrusted
    input, so it is checked here and answered with a 400.
    """
    if not isinstance(definition, dict):
        raise ServiceError("Question %r must be an object" % (qid,))
    kind = definition.get("type")
    if kind not in QUESTION_TYPES:
        raise ServiceError(
            "Question %r has unknown type %r; expected one of %s"
            % (qid, kind, ", ".join(QUESTION_TYPES))
        )
    if not isinstance(definition.get("instructions"), str):
        raise ServiceError("Question %r must have a string 'instructions'" % (qid,))
    criteria = definition.get("criteria")
    if kind == "choice":
        labels = list(criteria) if isinstance(criteria, (dict, list)) else []
        if not labels:
            raise ServiceError("Question %r is a choice with no criteria" % (qid,))
        if not all(isinstance(label, str) for label in labels):
            raise ServiceError("Question %r has non-string choice labels" % (qid,))
        if len(set(labels)) != len(labels):
            raise ServiceError("Question %r has duplicate choice labels" % (qid,))
    elif kind == "score":
        if not isinstance(criteria, list) or not criteria:
            raise ServiceError("Question %r is a score with no criteria list" % (qid,))
        if not all(isinstance(level, str) for level in criteria):
            raise ServiceError("Question %r has non-string score levels" % (qid,))


class Handler(BaseHTTPRequestHandler):
    registry = None
    server_version = "laya-mlx-service/%s" % __version__
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        print("[http] %s %s" % (self.address_string(), format % args), flush=True)

    # ------------------------------------------------------------------ plumbing
    def _send(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError as error:
            raise ServiceError("Invalid Content-Length header") from error
        if length < 0 or length > MAX_BODY_BYTES:
            raise ServiceError("Request body must be between 0 and %d bytes" % MAX_BODY_BYTES)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ServiceError("Invalid JSON body: %s" % error) from error

    def _slot(self):
        return _InferenceSlot(self.registry)

    def _questions(self, body):
        questions = body.get("questions")
        if questions is None:
            factory = PRESETS.get(body.get("preset"))
            if factory is not None:
                return factory()
        if not isinstance(questions, dict) or not questions:
            raise ServiceError("Provide a nonempty 'questions' object or a known 'preset'")
        for qid, definition in questions.items():
            validate_question(qid, definition)
        return questions

    def _state(self, body):
        state = body.get("state")
        if state is None:
            raise ServiceError("Provide 'state' (text, dict, or conversation list)")
        return state

    def _predict_with(self, name, state, questions):
        agent = self.registry.get(name)
        started = time.perf_counter()
        with self._slot():
            result = agent.predict(state, questions)
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        result["checkpoint"] = name
        return result

    # ------------------------------------------------------------------ verbs
    def do_GET(self):
        route = self.path.split("?", 1)[0].rstrip("/") or "/"
        if route == "/health":
            self._send(
                200,
                {
                    "status": "ok",
                    "version": __version__,
                    "resident": self.registry.resident,
                    "checkpoints": {
                        name: describe(path)
                        for name, path in sorted(self.registry.checkpoints.items())
                    },
                },
            )
        elif route == "/presets":
            self._send(200, {"presets": sorted(PRESETS)})
        else:
            self._send(404, {"error": "not found", "path": self.path})

    def do_POST(self):
        route = self.path.split("?", 1)[0].rstrip("/")
        try:
            body = self._read_json()
            if route == "/predict":
                self._send(200, self._predict(body))
            elif route == "/route":
                self._send(200, self._route(body))
            elif route == "/shortlist":
                self._send(200, self._shortlist(body))
            else:
                self._send(404, {"error": "not found", "path": self.path})
        except BusyError as error:
            self._send(503, {"error": str(error)})
        except CheckpointError as error:
            self._send(503, {"error": str(error)})
        except ServiceError as error:
            self._send(400, {"error": str(error)})
        except (ValueError, FloatingPointError) as error:
            self._send(400, {"error": "%s: %s" % (type(error).__name__, error)})
        except Exception as error:  # noqa: BLE001 - report, never kill the server
            self._send(500, {"error": "%s: %s" % (type(error).__name__, error)})

    # ------------------------------------------------------------------ handlers
    def _predict(self, body):
        name = body.get("checkpoint")
        if name is None:
            name, _ = self.registry.choose(self._state(body))
        return self._predict_with(name, self._state(body), self._questions(body))

    def _route(self, body):
        state = self._state(body)
        forced = body.get("model")
        name, reason = (
            (forced, "explicit model in request") if forced else self.registry.choose(state)
        )
        result = self._predict_with(name, state, self._questions(body))
        from ..lang import analyse

        return {**result, "routing_reason": reason, "language": analyse(state)}

    def _shortlist(self, body):
        name = body.get("checkpoint")
        if name is None:
            name, _ = self.registry.choose(self._state(body))
        state = self._state(body)
        labels = body.get("labels")
        if not isinstance(labels, list) or not labels:
            raise ServiceError("Provide a nonempty 'labels' list of strings")
        if not all(isinstance(label, str) for label in labels):
            raise ServiceError("Every entry in 'labels' must be a string")
        questions = {
            "choice": {
                "type": "choice",
                "instructions": body.get("instructions", "Which option best applies?"),
                "criteria": list(labels),
            }
        }
        agent = self.registry.get(name)
        embed_fn = embed_fn_from_agent(agent)
        started = time.perf_counter()
        with self._slot():
            result = predict_shortlist(agent, state, questions, embed_fn, k=int(body.get("k", 20)))
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        result["checkpoint"] = name
        return result


class _InferenceSlot:
    """Bound in-flight GPU work; over-capacity callers wait, then get a 503."""

    def __init__(self, registry):
        self._registry = registry

    def __enter__(self):
        if not self._registry.acquire_slot():
            raise BusyError(
                "Server busy: all %d inference slots occupied for %ds"
                % (self._registry.max_inflight, self._registry.queue_timeout)
            )
        return self

    def __exit__(self, *_):
        self._registry.release_slot()
        return False


class Server(ThreadingHTTPServer):
    """ThreadingHTTPServer sized for real client bursts.

    socketserver defaults `request_queue_size` to 5, so a burst above that is
    reset at the TCP layer before any application code runs. 128 is a normal
    HTTP listen backlog.
    """

    request_queue_size = 128
    daemon_threads = True
    allow_reuse_address = True


def build_server(host="127.0.0.1", port=8080, checkpoints=None, **registry_kwargs):
    """Create a configured (unstarted) server bound to a Registry."""
    handler = type(
        "BoundHandler", (Handler,), {"registry": Registry(checkpoints, **registry_kwargs)}
    )
    return Server((host, port), handler)
