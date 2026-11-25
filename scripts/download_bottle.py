#!/usr/bin/env python3
"""Download only the MVTec AD ``bottle`` category from an open HF mirror.

The official MVTec AD archive is ~5.3 GB; this script fetches just the 357
files needed for the bottle subset (~180 MB) from ``foersben/mvtec-ad``.

Example:
    python scripts/download_bottle.py --out data/mvtec_anomaly_detection
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPO = "foersben/mvtec-ad"
API = "https://hf-mirror.com/api/datasets/" + REPO
RESOLVE = "https://hf-mirror.com/datasets/" + REPO + "/resolve/main/"
HEADERS = {"User-Agent": "Mozilla/5.0 (defect-detection-project)"}


def list_bottle_files() -> list[str]:
    response = requests.get(API, timeout=60, headers=HEADERS)
    response.raise_for_status()
    siblings = response.json()["siblings"]
    return sorted(
        item["rfilename"]
        for item in siblings
        if item["rfilename"].startswith("bottle/")
    )


def download_one(rel_path: str, out_root: Path, retries: int = 3) -> int:
    target = out_root / rel_path
    if target.exists() and target.stat().st_size > 0:
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    url = RESOLVE + rel_path + "?download=true"
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=180, headers=HEADERS)
            response.raise_for_status()
            target.write_bytes(response.content)
            return len(response.content)
        except Exception as exc:  # noqa: BLE001 - retry transient network errors
            last_error = exc
            time.sleep(1 + attempt)
    raise RuntimeError(f"failed to download {rel_path}: {last_error}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "mvtec_anomaly_detection")
    parser.add_argument("--workers", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    files = list_bottle_files()
    print(f"[info] {len(files)} files in bottle subset -> {args.out}", flush=True)

    downloaded = 0
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, f, args.out): f for f in files}
        for done, future in enumerate(as_completed(futures), start=1):
            total_bytes += future.result()
            downloaded += 1
            if done % 25 == 0 or done == len(files):
                print(
                    f"[progress] {done}/{done} files, {total_bytes / 1e6:.1f} MB",
                    flush=True,
                )
    print(f"[done] bottle subset ready at {args.out / 'bottle'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
