"""Download, validate and wrangle the attributed UCI hourly bike dataset."""
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import hashlib
import json
import numpy as np
import pandas as pd
import requests

URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
FEATURES = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed", "hour_sin", "hour_cos", "month_sin", "month_cos"]

def download(directory="data"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "hour.csv"
    if not path.exists():
        response = requests.get(URL, timeout=60)
        response.raise_for_status()
        with ZipFile(BytesIO(response.content)) as archive:
            member = next(n for n in archive.namelist() if n.endswith("hour.csv"))
            path.write_bytes(archive.read(member))
        (directory / "provenance.json").write_text(json.dumps({"url": URL, "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "license": "CC BY 4.0", "citation": "Fanaee-T, H. (2013). Bike Sharing. DOI:10.24432/C5W894"}, indent=2))
    return pd.read_csv(path)

def wrangle(raw):
    missing = set(FEATURES[:12] + ["dteday", "cnt"]) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = raw.copy()
    df["timestamp"] = pd.to_datetime(df["dteday"], errors="coerce") + pd.to_timedelta(pd.to_numeric(df["hr"], errors="coerce"), unit="h")
    for column in FEATURES[:12] + ["cnt"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["timestamp", "cnt", "hr", "mnth"])
    df = df[df["cnt"].ge(0) & df["hr"].between(0, 23) & df["mnth"].between(1,12)]
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    for name, period, prefix in [("hr", 24, "hour"), ("mnth", 12, "month")]:
        df[prefix + "_sin"] = np.sin(2 * np.pi * df[name] / period)
        df[prefix + "_cos"] = np.cos(2 * np.pi * df[name] / period)
    # casual + registered equals cnt: never include these leakage columns.
    return df
