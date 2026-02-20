"""CLI entrypoint wrapper for model comparison.

Core comparison/reporting behavior lives in `src.scoring.compare_models`.
"""

from __future__ import annotations

from .compare_models import main


if __name__ == "__main__":
    main()
