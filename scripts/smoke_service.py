"""End-to-end checks for the laya-mlx HTTP service.

    .venv/bin/python scripts/smoke_service.py --base http://127.0.0.1:8090

Exits non-zero on the first failed check so it can gate a deploy.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

FAILURES = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {name}{(' -> ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(name)
    return condition


def request(base, path, payload=None, method=None):
    url = base + path
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method=method
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", "replace")
        try:
            return error.code, json.loads(body)
        except json.JSONDecodeError:
            return error.code, {"raw": body}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8080")
    parser.add_argument("--concurrency", type=int, default=200)
    args = parser.parse_args()
    base = args.base.rstrip("/")

    status, health = request(base, "/health")
    check("GET /health -> 200", status == 200, str(status))
    check("health reports ok", health.get("status") == "ok")
    check(
        "health lists checkpoints",
        bool(health.get("checkpoints")),
        str(list(health.get("checkpoints", {}))),
    )

    status, presets = request(base, "/presets")
    check("GET /presets -> 200", status == 200, str(status))
    check("presets include triage", "triage" in presets.get("presets", []))

    # Preset triage on Chinese input.
    status, result = request(
        base, "/predict", {"state": {"message": "重复扣款,请退款。"}, "preset": "triage"}
    )
    check("POST /predict preset -> 200", status == 200, str(status))
    answers = result.get("answers", {})
    check(
        "triage intent == refund",
        answers.get("intent", {}).get("choice") == "refund",
        str(answers.get("intent", {}).get("choice")),
    )
    check(
        "triage refund_requested > 0.8",
        answers.get("refund_requested", {}).get("noul", 0) > 0.8,
        str(answers.get("refund_requested", {}).get("noul")),
    )
    check(
        "predict reports latency_ms",
        isinstance(result.get("latency_ms"), (int, float)),
        str(result.get("latency_ms")),
    )
    check(
        "predict reports checkpoint",
        result.get("checkpoint") == "multilingual",
        str(result.get("checkpoint")),
    )

    # All three question types via custom questions.
    status, result = request(
        base,
        "/predict",
        {
            "state": "The payment API times out above 10 MB request bodies.",
            "checkpoint": "english",
            "questions": {
                "team": {
                    "type": "choice",
                    "instructions": "Which team owns this?",
                    "criteria": {
                        "backend": "API internals",
                        "frontend": "client UI",
                        "devops": "infrastructure",
                    },
                },
                "severity": {
                    "type": "score",
                    "instructions": "How severe?",
                    "criteria": ["cosmetic", "degraded", "outage"],
                },
                "human": {"type": "noul", "instructions": "Does a human need to look at it?"},
            },
        },
    )
    check("custom questions -> 200", status == 200, str(status))
    a = result.get("answers", {})
    check(
        "choice type answered",
        a.get("team", {}).get("choice") == "backend",
        str(a.get("team", {}).get("choice")),
    )
    check(
        "score is in range",
        0 <= a.get("severity", {}).get("score", -1) <= 2,
        str(a.get("severity", {}).get("score")),
    )
    check(
        "noul in [0,1]",
        0 <= a.get("human", {}).get("noul", -1) <= 1,
        str(a.get("human", {}).get("noul")),
    )

    # Routing: script decides the checkpoint.
    for text, expected, label in [
        ("发票被重复扣款,请退款。", "multilingual", "Han script"),
        ("I was double charged and want a refund.", "english", "English Latin"),
        ("Die Rechnung wurde doppelt belastet.", "multilingual", "German"),
    ]:
        status, routed = request(
            base,
            "/route",
            {"state": text, "questions": {"r": {"type": "noul", "instructions": "Refund?"}}},
        )
        check(
            "route %s -> %s" % (label, expected),
            status == 200 and routed.get("checkpoint") == expected,
            str(routed.get("checkpoint")),
        )
        check(
            "route %s explains itself" % label,
            bool(routed.get("routing_reason")),
            str(routed.get("routing_reason")),
        )

    # Shortlist: large label set reduced to top-k.
    labels = [
        "billing_refund",
        "billing_invoice",
        "tech_outage",
        "tech_login",
        "sales_upgrade",
        "sales_demo",
        "security_phish",
        "hr_payroll",
    ]
    status, result = request(
        base,
        "/shortlist",
        {
            "state": "A customer cannot log in after their password reset email bounced.",
            "labels": labels,
            "k": 3,
            "instructions": "Which tag best classifies this?",
        },
    )
    check("shortlist -> 200", status == 200, str(status))
    meta = result.get("shortlist", {}).get("choice", {})
    check("shortlist kept k labels", len(meta.get("labels", [])) == 3, str(meta.get("labels")))
    check(
        "shortlist probs over kept only",
        set(result.get("answers", {}).get("choice", {}).get("probabilities", {}))
        == set(meta.get("labels", [])),
    )

    # Error handling.
    for name, payload, want in [
        ("missing state", {"questions": {"a": {"type": "noul", "instructions": "x"}}}, 400),
        (
            "bad question type",
            {"state": "x", "questions": {"a": {"type": "bogus", "instructions": "x"}}},
            400,
        ),
        ("empty questions", {"state": "x", "questions": {}}, 400),
        (
            "unknown checkpoint",
            {
                "state": "x",
                "questions": {"a": {"type": "noul", "instructions": "y"}},
                "checkpoint": "nope",
            },
            503,
        ),
    ]:
        status, _ = request(base, "/predict", payload)
        check("error: %s -> %d" % (name, want), status == want, str(status))

    status, _ = request(base, "/nope", {})
    check("unknown route -> 404", status == 404, str(status))

    # Concurrency: a burst must not reset connections.
    def call(i):
        _, r = request(
            base,
            "/predict",
            {
                "state": "service is down and customers are angry",
                "questions": {"a": {"type": "noul", "instructions": "Is it urgent?"}},
            },
        )
        return r["answers"]["a"]["noul"]

    workers = 32
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        values = list(pool.map(call, range(args.concurrency)))
    wall = time.perf_counter() - started
    check("concurrent burst completed", len(values) == args.concurrency, str(len(values)))
    check("concurrent results deterministic", len(set(values)) == 1, str(set(values)))
    print(
        "[info] %d requests / %d workers in %.2fs (%.0f req/s)"
        % (args.concurrency, workers, wall, args.concurrency / wall)
    )

    if FAILURES:
        print("%d checks failed: %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
