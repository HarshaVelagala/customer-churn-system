"""Application entrypoint for initialization and app launching."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.initialize import initialize


def main() -> None:
    initialize()


if __name__ == "__main__":
    main()
