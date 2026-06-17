"""Shared pytest setup for the core-logic suite.

Two concerns are handled here:

1. Import path. The tests import the app's own packages by their absolute
   names (``engine.*`` / ``gui.*``). ``python -m pytest`` already puts the
   repo root on ``sys.path``, but inserting it explicitly also lets the suite
   run under a bare ``pytest tests/``.

2. Headless Qt. ``gui.model_download_manager`` imports ``PyQt6.QtCore``; the
   offscreen platform plugin lets it load on a CI runner with no display.
"""

import os
import sys
from pathlib import Path

# Repo root = the directory that contains this tests/ folder.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
