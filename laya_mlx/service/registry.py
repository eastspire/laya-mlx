"""Shared, lazily loaded Agents and script-based checkpoint selection.

Loading is guarded by a re-entrant lock so concurrent requests share one Agent
instead of racing to build duplicates. Inference is not serialised: MLX queues
GPU work itself, so callers are bounded by a semaphore rather than a lock.
"""

import threading
import time

from ..agent import load as load_agent
from ..lang import analyse
from .checkpoint import CheckpointError, resolve_checkpoint

ENGLISH = "english"
MULTILINGUAL = "multilingual"


class Registry:
    """One Agent per checkpoint, created on first use and shared afterwards."""

    def __init__(
        self,
        checkpoints=None,
        dtype="float16",
        compile=True,
        cache_prompts=True,
        max_inflight=16,
        queue_timeout=60.0,
    ):
        self.dtype = dtype
        self.compile = compile
        self.cache_prompts = cache_prompts
        self.max_inflight = max_inflight
        self.queue_timeout = queue_timeout
        self.checkpoints = {k: resolve_checkpoint(v) for k, v in (checkpoints or {}).items()}
        self._agents = {}
        self._lock = threading.Lock()
        self._inflight = threading.Semaphore(max_inflight)

    @property
    def resident(self):
        with self._lock:
            return sorted(self._agents)

    def get(self, name):
        agent = self._agents.get(name)
        if agent is not None:
            return agent
        with self._lock:
            if name not in self._agents:
                path = self.checkpoints.get(name)
                if path is None:
                    raise CheckpointError(
                        "Unknown checkpoint %r; available: %s"
                        % (name, ", ".join(sorted(self.checkpoints)) or "none")
                    )
                started = time.perf_counter()
                try:
                    self._agents[name] = load_agent(
                        str(path),
                        dtype=self.dtype,
                        compile=self.compile,
                        cache_prompts=self.cache_prompts,
                    )
                except FileNotFoundError as error:
                    raise CheckpointError(str(error)) from error
                print(
                    "[registry] loaded %s in %.0f ms"
                    % (name, (time.perf_counter() - started) * 1000),
                    flush=True,
                )
            return self._agents[name]

    def acquire_slot(self, timeout=None):
        """Return a token holding one inference slot, or None when saturated."""
        return self._inflight.acquire(timeout=self.queue_timeout if timeout is None else timeout)

    def release_slot(self, _token=None):
        self._inflight.release()

    def choose(self, state):
        """Pick a checkpoint from the input's script, mirroring the library Router.

        The English checkpoint uses an English BPE vocabulary and collapses to
        near-random on scripts it cannot read, so script detection is the primary
        signal. Latin-script language identification is a stopword heuristic and
        is deliberately best-effort.
        """
        evidence = analyse(state)
        script = evidence.get("script")
        if script == "unknown":
            return self._fallback(MULTILINGUAL, "no letters detected")
        if script != "latin":
            return self._fallback(
                MULTILINGUAL,
                "non-Latin script (%s, %.0f%% of letters); the English checkpoint "
                "cannot read it" % (script, 100 * float(evidence["non_latin_fraction"])),
            )
        if not evidence.get("is_english"):
            language = evidence.get("language")
            if language:
                return self._fallback(
                    MULTILINGUAL, "Latin script but language looks like %r" % language
                )
            return self._fallback(
                MULTILINGUAL,
                "Latin script, language unidentified, %.0f%% non-English letters"
                % (100 * float(evidence["diacritic_rate"])),
            )
        return self._fallback(ENGLISH, "English Latin text")

    def _fallback(self, preferred, reason):
        """Keep the preferred checkpoint when present, else the first available."""
        if preferred in self.checkpoints:
            return preferred, reason
        available = sorted(self.checkpoints)
        if not available:
            raise CheckpointError("No checkpoints configured")
        return available[0], "%s (fallback: %s unavailable)" % (reason, preferred)
