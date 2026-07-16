"""pytest fixtures shared across test modules.

Ensures the project root is importable and that OPENAI_API_KEY is set
to *something* before ``agent.py`` is imported, so ChatOpenAI does not
complain even though our tests never actually call it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-used")
