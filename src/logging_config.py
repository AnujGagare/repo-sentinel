"""
Structured logging setup.

Replaces scattered print() statements (which don't carry log level,
timestamp, or module context, and can't be filtered/redirected) with
Python's standard logging module, configured once at app startup.
"""

from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers if this is called more than once (e.g.
    # under uvicorn's --reload, which re-imports the app module).
    if root.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(handler)

    # Chroma and httpx are noisy at INFO -- keep them at WARNING unless
    # someone explicitly wants DEBUG for the whole app.
    if level != "DEBUG":
        logging.getLogger("chromadb").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
