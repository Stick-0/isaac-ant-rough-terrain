#!/usr/bin/env python3
"""Apply the published changes to a clean, pinned IsaacLab_RS checkout."""

import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("isaaclab", type=Path)
    parser.add_argument("--check", action="store_true", help="Validate without changing files.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "manifest.json").read_text())
    lab = args.isaaclab.resolve()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=lab, text=True).strip()
    if head != manifest["base_commit"]:
        raise SystemExit(f"Expected IsaacLab_RS commit {manifest['base_commit']}, found {head}")
    patch = root / "patches/isaaclab.patch"
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=lab, check=True)
    if not args.check:
        subprocess.run(["git", "apply", str(patch)], cwd=lab, check=True)
    print("Patch validated." if args.check else "Ant terrain and stability changes applied.")


if __name__ == "__main__":
    main()
