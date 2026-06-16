"""
Just some progress loging staff used when verbose is set to 1 in the solver
"""

from __future__ import annotations

import sys
import time


class ProgressLogger:
    """Hierarchical \r-based progress display for nested pipeline levels."""

    def __init__(self, verbose: int):
        self.verbose = verbose
        self.level   = -1

    def enter_level(self, N: int, D: int, N_ITER: int, reason: str = "") -> _LevelCtx:
        """Push a new recursion level and return its context."""
        self.level += 1
        return _LevelCtx(self, N, D, N_ITER, reason)

    def exit_level(self) -> None:
        self.level -= 1

    def _prefix(self) -> str:
        return f"[L{self.level}] "

    def write(self, msg: str, newline: bool = False) -> None:
        if self.verbose < 1:
            return
        sys.stdout.write(f"\r{self._prefix()}{msg}    ")
        if newline:
            sys.stdout.write("\n")
        sys.stdout.flush()


class _LevelCtx:
    """Tracks timing and tick state for a single pipeline level."""

    def __init__(self, logger: ProgressLogger, N: int, D: int, N_ITER: int, reason: str = ""):
        self._log    = logger
        self.N       = N
        self.D       = D
        self.N_ITER  = N_ITER
        self.reason  = reason
        self._tick   = 0
        self._t0: float | None     = None
        self._t_iter: float | None = None

    def on_compile(self) -> None:
        self._log.write("compiling JAX kernel…")

    def on_bruteforce_start(self) -> None:
        suffix = f"  ({self.reason})" if self.reason else ""
        self._log.write(f"bruteforce  N={self.N}  D={self.D}{suffix} …")

    def on_bruteforce_done(self) -> None:
        self._log.write("bruteforce done ✓", newline=True)

    def tick(self) -> None:
        """Called once per gridification callback (= one full iteration). Drives the ETA display."""
        now = time.perf_counter()
        self._tick += 1

        if self._tick == 1:
            self._t0 = now
            self._log.write(f"{self._bar()} — calibrating…")
            return

        if self._tick == 2:
            self._t_iter = now - self._t0  # type: ignore[operator]

        self._log.write(f"{self._bar()} — {self._eta(now)} remaining")

    def done(self) -> None:
        self._log.write(f"{self._bar(done=True)} — done ✓", newline=True)

    def _bar(self, done: bool = False) -> str:
        filled = self.N_ITER if done else max(0, self._tick - 1)
        W      = 20
        n_fill = int(W * filled / self.N_ITER)
        bar    = "▓" * n_fill + "░" * (W - n_fill)
        return f"{filled}/{self.N_ITER} [{bar}]"

    def _eta(self, now: float) -> str:
        if self._t_iter is None:
            return "?"
        remaining = (self.N_ITER - (self._tick - 1)) * self._t_iter
        return f"~{remaining:.0f}s" if remaining < 60 else f"~{remaining / 60:.1f}min"