"""Python entry point for DVC that also works in an unactivated Windows venv.

    python -m src.manage_data build
    python -m src.manage_data status

Generated CSVs can also be recreated without the DVC CLI wrapper:
``python -m src.data_pipeline all``.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build", "status", "push"])
    args = parser.parse_args()
    commands = {"build": ["repro"], "status": ["status"], "push": ["push"]}
    environment = os.environ.copy()
    environment.setdefault("DVC_SITE_CACHE_DIR", str(ROOT / ".dvc/site-cache"))
    environment["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    )
    subprocess.run(
        [sys.executable, "-m", "dvc", *commands[args.action]],
        cwd=ROOT,
        env=environment,
        check=True,
    )


if __name__ == "__main__":
    main()
