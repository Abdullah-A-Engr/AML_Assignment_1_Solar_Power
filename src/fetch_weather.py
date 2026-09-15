from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

LAT = 14.82
LON = 78.28
START = "2020-05-15"
END = "2020-06-17"
TZ = "Asia/Kolkata"

def main():
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": START,
        "end_date": END,
        "hourly": "shortwave_radiation,temperature_2m,cloud_cover",
        "timezone": TZ,
    }

    print("Requesting Open-Meteo historical weather...")
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()

    hourly = payload["hourly"]
    weather = pd.DataFrame(
        {
            "datetime": pd.to_datetime(hourly["time"]),
            "sw radiation": hourly["shortwave_radiation"],
            "temp 2m": hourly["temperature_2m"],
            "cloud cover": hourly["cloud_cover"],
        }
    )

    out = DATA / "plant1_openmeteo.csv"
    weather.to_csv(out, index=False)
    print(f"Saved {len(weather)} Open-Meteo rows to {out}")
    print("Timezone:", payload.get("timezone"))
    print("Coordinates returned:", payload.get("latitude"), payload.get("longitude"))

if __name__ == "__main__":
    main()
