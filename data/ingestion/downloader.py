"""Corpus downloader for open-india-law dataset by Vaquill-AI.

Fetches central legislation Parquet (~26MB) containing normalized India Code sections
down to the individual provision level, with official government source URLs.
"""

import os
from pathlib import Path
from typing import Optional
import requests
from tqdm import tqdm

from solvemycase.config.settings import get_settings


def download_central_legislation(
    target_path: Optional[Path] = None,
    force_download: bool = False
) -> Path:
    """Download the official in_central_legislation.parquet from open-india-law mirror.

    Args:
        target_path: Optional destination Path. Defaults to settings.data_cache_dir / in_central_legislation.parquet.
        force_download: If True, re-downloads even if the file already exists locally.

    Returns:
        Path to the downloaded parquet file.

    Raises:
        RuntimeError: If download fails or returns non-200 HTTP status.
    """
    settings = get_settings()
    dest_dir = settings.resolve_path(settings.data_cache_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if target_path is None:
        target_path = dest_dir / "in_central_legislation.parquet"
    else:
        target_path = Path(target_path)

    if target_path.exists() and not force_download:
        file_size_mb = target_path.stat().st_size / (1024 * 1024)
        print(f"[Corpus] in_central_legislation.parquet already exists at {target_path} ({file_size_mb:.2f} MB). Skipping download.")
        return target_path

    url = settings.legislation_source_url
    print(f"[Corpus] Downloading central legislation corpus from: {url}")

    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total_bytes = int(response.headers.get("content-length", 0))
        temp_target = target_path.with_suffix(".tmp")

        with open(temp_target, "wb") as f, tqdm(
            desc="Downloading legislation.parquet",
            total=total_bytes,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 64):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

        temp_target.replace(target_path)
        print(f"[Corpus] Successfully saved to: {target_path} ({target_path.stat().st_size / (1024 * 1024):.2f} MB)")
        return target_path

    except Exception as exc:
        if "temp_target" in locals() and temp_target.exists():
            temp_target.unlink()
        raise RuntimeError(f"Failed to download central legislation from {url}: {exc}") from exc


if __name__ == "__main__":
    download_central_legislation()
