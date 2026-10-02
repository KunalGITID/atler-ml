"""Call ATLER's TypeScript core from Python (see bridge/atler.ts)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "bridge" / "atler.ts"


def atler_dir() -> Path:
    return Path(os.environ.get("ATLER_DIR", ROOT.parent / "ATLER")).resolve()


def run(jobs: list[dict[str, Any]]) -> list[Any]:
    """Run a batch of jobs in one Node process; results come back in order."""
    out = subprocess.run(
        ["node", str(SCRIPT)],
        input=json.dumps(jobs),
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "ATLER_DIR": str(atler_dir())},
    )
    if out.returncode != 0:
        raise RuntimeError(f"ATLER bridge failed:\n{out.stderr}")
    return json.loads(out.stdout)
