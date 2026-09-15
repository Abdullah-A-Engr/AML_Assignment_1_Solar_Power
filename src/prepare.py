from pathlib import Path
import sys
import re
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import load_raw

def inspect_and_parse_datetime(series, label):
    """Inspect raw strings first, then parse using the detected format."""
    sample = series.dropna().astype(str).head(5).tolist()
    print(f"\n{label} raw DATE_TIME samples:")
    for value in sample:
        print(" ", value)

    first = sample[0]
    if re.match(r"^\d{2}-\d{2}-\d{4} \d{2}:\d{2}$", first):
        fmt = "%d-%m-%Y %H:%M"
    elif re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", first):
        fmt = "%Y-%m-%d %H:%M:%S"
    elif re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$", first):
        fmt = "%Y-%m-%d %H:%M"
    else:
        # Fall back only after inspecting the raw values.
        fmt = None

    parsed = pd.to_datetime(series, format=fmt, errors="coerce")
    if parsed.isna().any():
        # A second explicit pass handles files containing a consistent alternate format.
        parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)

    if parsed.isna().any():
        bad = int(parsed.isna().sum())
        raise ValueError(f"{label}: {bad} timestamps could not be parsed.")
    return parsed

def main():
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    raw = load_raw(DATA)

    gen = raw["gen1"].copy()
    sensor = raw["sensor1"].copy()

    gen["datetime"] = inspect_and_parse_datetime(gen["DATE_TIME"], "Plant 1 generation")
    sensor["datetime"] = inspect_and_parse_datetime(sensor["DATE_TIME"], "Plant 1 sensor")

    print("\nConverted timestamp ranges:")
    print("Generation:", gen["datetime"].min(), "to", gen["datetime"].max())
    print("Sensor:    ", sensor["datetime"].min(), "to", sensor["datetime"].max())

    # Plant-level power = sum across all inverters at each timestamp.
    power = (
        gen.groupby("datetime", as_index=False)[["AC_POWER", "DC_POWER"]]
        .sum()
        .rename(columns={"AC_POWER": "ac power", "DC_POWER": "dc power"})
    )

    sensors = (
        sensor.groupby("datetime", as_index=False)[
            ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]
        ]
        .mean()
        .rename(
            columns={
                "AMBIENT_TEMPERATURE": "ambient temp",
                "MODULE_TEMPERATURE": "module temp",
                "IRRADIATION": "irradiation",
            }
        )
    )

    # Outer merge is intentional so timestamps existing in only one file can be counted.
    merged = pd.merge(power, sensors, on="datetime", how="outer", indicator=True)
    only_one = int((merged["_merge"] != "both").sum())
    only_generation = int((merged["_merge"] == "left_only").sum())
    only_sensor = int((merged["_merge"] == "right_only").sum())

    print(f"\nTimestamps in only one file: {only_one}")
    print(f"  Generation only: {only_generation}")
    print(f"  Sensor only:     {only_sensor}")

    merged = merged.drop(columns="_merge").set_index("datetime").sort_index()

    # The assignment asks for hourly means.
    hourly = merged.resample("1h").mean()

    missing_rows = int(hourly.isna().any(axis=1).sum())
    missing_cells = int(hourly.isna().sum().sum())
    print(f"\nHourly rows: {len(hourly)}")
    print(f"Hourly rows containing missing values: {missing_rows}")
    print(f"Missing cells: {missing_cells}")

    # Missing values are handled by linear interpolation only if any are present,
    # then edge values are forward/back filled. Normally this should be unnecessary.
    if missing_cells:
        hourly = hourly.interpolate(method="time").ffill().bfill()
        print("Handling: time interpolation, followed by forward/back fill for edge gaps.")
    else:
        print("Handling: no missing hourly values were present; no imputation applied.")

    hourly = hourly.reset_index()
    out = DATA / "plant1_hourly.csv"
    hourly.to_csv(out, index=False)

    prep = pd.DataFrame(
        {
            "metric": [
                "Raw generation rows (Plant 1)",
                "Raw sensor rows (Plant 1)",
                "Timestamps in only one file",
                "Generation-only timestamps",
                "Sensor-only timestamps",
                "Hourly rows after resampling",
                "Hourly rows with missing values before handling",
                "Missing cells before handling",
            ],
            "value": [
                len(gen),
                len(sensor),
                only_one,
                only_generation,
                only_sensor,
                len(hourly),
                missing_rows,
                missing_cells,
            ],
        }
    )
    prep.to_csv(RESULTS / "table1_data_preparation.csv", index=False)
    print(f"\nSaved {out}")

if __name__ == "__main__":
    main()
