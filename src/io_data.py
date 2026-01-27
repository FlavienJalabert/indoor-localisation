"""Dataset download and loading utilities."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests


def download_zip(url: str, timeout_s: int = 60) -> bytes:
    """Download the dataset ZIP from a URL and return raw bytes."""

    resp = requests.get(url, timeout=timeout_s)
    resp.raise_for_status()
    return resp.content


def extract_csv_from_zip(zip_bytes: bytes, output_dir: Path) -> List[Path]:
    """Extract all CSV files from ZIP bytes into output_dir."""

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_paths: List[Path] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        for name in csv_names:
            extracted_path = zf.extract(name, path=output_dir)
            csv_paths.append(Path(extracted_path))
    return csv_paths


def load_csvs_to_dataframes(csv_paths: List[Path]) -> Dict[str, pd.DataFrame]:
    """Load each CSV into a DataFrame keyed by stem name."""

    dfs: Dict[str, pd.DataFrame] = {}
    for path in csv_paths:
        name_no_ext = path.stem
        dfs[name_no_ext] = pd.read_csv(path)
    return dfs


def load_or_download_dataset(
    zip_url: str, raw_dir: Path, *, force: bool = False
) -> Dict[str, pd.DataFrame]:
    """Load CSVs from raw_dir or download/extract from zip_url."""

    raw_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = list(raw_dir.rglob("*.csv"))
    if csv_paths and not force:
        return load_csvs_to_dataframes(csv_paths)

    zip_bytes = download_zip(zip_url)
    csv_paths = extract_csv_from_zip(zip_bytes, raw_dir)
    return load_csvs_to_dataframes(csv_paths)
